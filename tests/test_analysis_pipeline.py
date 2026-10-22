"""Pruebo la fachada del análisis con descarga y analizador falsos y la verificación de duración real."""

import asyncio
import shutil
import time
from pathlib import Path

import pytest
from kinesiapp.common.v1.errors_pb2 import ErrorReason

from kinesiapp_ai.analysis.frame_sampler import FrameSampler
from kinesiapp_ai.analysis.errors import VideoTooLongError
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage
from kinesiapp_ai.analysis.movement_strategy import Movement
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.pipeline.analysis_pipeline import MovementAnalysisPipeline, Progress
from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.pipeline.threaded_analysis import ThreadedAnalysisRunner
from kinesiapp_ai.rpc.domain_error import DomainError
from tests.video_fixtures import write_video

RESULT = object()
URL = "https://storage.neon.test/videos/a.mp4"


class _FileSource:
    """Copio un video sintético al destino como si viniera del bucket, y anoto dónde lo dejé."""

    def __init__(self, video: Path, error: Exception | None = None) -> None:
        self.video, self.error = video, error
        self.destinations: list[Path] = []

    def validate(self, url: str) -> None:
        if not url.startswith("https://storage.neon.test/"):
            raise DomainError(ErrorReason.ERROR_REASON_BAD_REQUEST, "Video host is not allowed")

    async def download(self, url: str, destination: Path) -> None:
        self.destinations.append(destination)
        shutil.copyfile(self.video, destination)
        if self.error:
            raise self.error


class _FakeAnalyzer:
    def __init__(self, frames: int, delay: float = 0.0) -> None:
        self.frames, self.delay = frames, delay

    def analyze(self, video_path: Path, strategies, observer):
        assert video_path.exists()
        observer.on_stage(AnalysisStage.EXTRACTING_POSE)
        for processed in range(1, self.frames + 1):
            time.sleep(self.delay)
            observer.on_frames(processed)
        observer.on_stage(AnalysisStage.SCORING_RISK)
        return RESULT


class _Builder:
    def __init__(self, analyzer: _FakeAnalyzer) -> None:
        self.analyzer = analyzer

    def build(self, models) -> _FakeAnalyzer:
        return self.analyzer


def _pipeline(
    source: _FileSource,
    analyzer: _FakeAnalyzer,
    slots: AnalysisSlots | None = None,
    load=lambda: "models",
    sampler: FrameSampler | None = None,
) -> MovementAnalysisPipeline:
    return MovementAnalysisPipeline(
        slots=slots or AnalysisSlots(1),
        warmer=ModelWarmer(load),
        analyzers=_Builder(analyzer),
        strategies={Movement.JUMP: "jump-strategy"},
        source=source,
        duration_guard=VideoDurationGuard(max_duration_seconds=2.0),
        runner=ThreadedAnalysisRunner(progress_every_frames=2),
        sampler=sampler or FrameSampler(None),
    )


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "short.mp4", frame_count=4, fps=10.0)


def test_the_stream_reports_each_stage_with_frames_and_ends_with_the_result(short_video: Path) -> None:
    source = _FileSource(short_video)

    async def main():
        return [event async for event in _pipeline(source, _FakeAnalyzer(frames=4)).analyze(URL, Movement.JUMP)]

    events = asyncio.run(main())

    assert events == [
        Progress(AnalysisStage.QUEUED),
        Progress(AnalysisStage.DOWNLOADING_VIDEO),
        Progress(AnalysisStage.EXTRACTING_POSE, 0, 4),
        Progress(AnalysisStage.EXTRACTING_POSE, 2, 4),
        Progress(AnalysisStage.EXTRACTING_POSE, 4, 4),
        Progress(AnalysisStage.SCORING_RISK),
        RESULT,
    ]
    assert not source.destinations[0].exists()


def test_with_a_frame_rate_cap_the_total_counts_only_the_sampled_frames(short_video: Path) -> None:
    async def main():
        pipeline = _pipeline(_FileSource(short_video), _FakeAnalyzer(frames=2), sampler=FrameSampler(max_fps=5))
        return [event async for event in pipeline.analyze(URL, Movement.JUMP)]

    totals = {event.frames_total for event in asyncio.run(main()) if isinstance(event, Progress) and event.frames_total}

    assert totals == {2}


def test_a_second_analysis_while_the_first_runs_is_rejected_at_once(short_video: Path) -> None:
    slots = AnalysisSlots(1)

    async def main():
        first = _pipeline(_FileSource(short_video), _FakeAnalyzer(frames=4), slots).analyze(URL, Movement.JUMP)
        await anext(first)
        try:
            with pytest.raises(DomainError) as raised:
                await anext(_pipeline(_FileSource(short_video), _FakeAnalyzer(frames=4), slots).analyze(URL, Movement.JUMP))
        finally:
            await first.aclose()
        return raised.value.reason, slots.in_use

    assert asyncio.run(main()) == (ErrorReason.ERROR_REASON_AI_BUSY, 0)


def test_a_disallowed_url_is_rejected_before_taking_a_slot_or_loading_models(short_video: Path) -> None:
    slots = AnalysisSlots(1)
    loads: list[int] = []

    async def main():
        pipeline = _pipeline(_FileSource(short_video), _FakeAnalyzer(frames=4), slots, load=lambda: loads.append(1))
        with slots.reserve(), pytest.raises(DomainError) as raised:
            await anext(pipeline.analyze("https://evil.test/a.mp4", Movement.JUMP))
        return raised.value.reason

    # Aunque el cupo esté lleno respondo BAD_REQUEST y no AI_BUSY: valido la URL primero
    assert asyncio.run(main()) == ErrorReason.ERROR_REASON_BAD_REQUEST
    assert loads == []


def test_a_failed_download_removes_the_file_and_frees_the_slot(short_video: Path) -> None:
    slots = AnalysisSlots(1)
    source = _FileSource(short_video, error=DomainError(ErrorReason.ERROR_REASON_STORAGE_UNAVAILABLE, "down"))

    async def main():
        async for _ in _pipeline(source, _FakeAnalyzer(frames=4), slots).analyze(URL, Movement.JUMP):
            pass

    with pytest.raises(DomainError):
        asyncio.run(main())

    assert not source.destinations[0].exists()
    assert slots.in_use == 0


def test_a_video_over_the_maximum_duration_is_rejected_before_analyzing(tmp_path: Path) -> None:
    source = _FileSource(write_video(tmp_path / "long.mp4", frame_count=25, fps=10.0))
    stages: list[AnalysisStage] = []

    async def main():
        async for event in _pipeline(source, _FakeAnalyzer(frames=25)).analyze(URL, Movement.JUMP):
            stages.append(event.stage)

    with pytest.raises(VideoTooLongError):
        asyncio.run(main())

    assert AnalysisStage.EXTRACTING_POSE not in stages
    assert not source.destinations[0].exists()


def test_cancelling_mid_analysis_removes_the_file_and_frees_the_slot(short_video: Path) -> None:
    slots = AnalysisSlots(1)
    source = _FileSource(short_video)
    pipeline = _pipeline(source, _FakeAnalyzer(frames=10_000, delay=0.001), slots)

    async def main():
        async def consume():
            async for _ in pipeline.analyze(URL, Movement.JUMP):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())

    assert not source.destinations[0].exists()
    assert slots.in_use == 0


def test_models_that_fail_to_load_answer_ai_unavailable(short_video: Path) -> None:
    def broken_load():
        raise OSError("model file missing")

    async def main():
        async for _ in _pipeline(_FileSource(short_video), _FakeAnalyzer(frames=4), load=broken_load).analyze(URL, Movement.JUMP):
            pass

    with pytest.raises(DomainError) as raised:
        asyncio.run(main())

    assert raised.value.reason == ErrorReason.ERROR_REASON_AI_UNAVAILABLE
