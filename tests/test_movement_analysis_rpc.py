"""Pruebo AnalyzeVideo por la red con todo real salvo el bucket (transporte falso) y MediaPipe (estimador falso)."""

import asyncio
import tempfile
import time
from functools import partial
from pathlib import Path

import grpc
import httpx
import numpy as np
import pytest
from kinesiapp.ai.v1 import movement_analysis_pb2 as analysis_pb2
from kinesiapp.ai.v1 import movement_analysis_pb2_grpc as analysis_pb2_grpc
from kinesiapp.common.v1 import movement_pb2

from kinesiapp_ai.analysis.frame_sampler import FrameSampler
from kinesiapp_ai.analysis.angles import AngleCalculator
from kinesiapp_ai.analysis.movement_analyzer import MovementAnalyzer
from kinesiapp_ai.analysis.pose_extraction import VideoPoseExtractor
from kinesiapp_ai.analysis.preprocessing import LandmarkSeriesPreprocessor
from kinesiapp_ai.analysis.risk_level import RiskLevelClassifier
from kinesiapp_ai.analysis.video_duration import VideoDurationGuard
from kinesiapp_ai.analysis.video_reader import VideoFrameReader
from kinesiapp_ai.lifecycle.model_warmer import ModelWarmer
from kinesiapp_ai.pipeline.analysis_pipeline import MovementAnalysisPipeline
from kinesiapp_ai.pipeline.analysis_slots import AnalysisSlots
from kinesiapp_ai.pipeline.strategy_factory import MovementStrategyFactory
from kinesiapp_ai.pipeline.threaded_analysis import ThreadedAnalysisRunner
from kinesiapp_ai.rpc.analysis_messages import AnalysisMessages
from kinesiapp_ai.rpc.movement_analysis_servicer import MovementAnalysisServicer
from kinesiapp_ai.settings.phases import MovementPhaseSettings
from kinesiapp_ai.settings.risk import RiskSettings
from kinesiapp_ai.settings.video_download import VideoDownloadSettings
from kinesiapp_ai.video_source.http_video_source import HttpVideoSource
from kinesiapp_ai.video_source.url_policy import AllowedHostPolicy
from tests.grpc_harness import WITH_TOKEN, error_reason, serving
from tests.pose_fixtures import FPS, add_jump, standing_points
from tests.video_fixtures import write_video

HOST = "storage.neon.test"
VIDEO_URL = f"https://{HOST}/videos/jump.mp4?X-Amz-Signature=abc"
MAX_SECONDS = 4.0
JUMP_FRAMES = 90


class _JumpEstimator:
    """Devuelvo, para cada frame del video, la pose de un salto sintético según su timestamp."""

    def __init__(self, delay: float = 0.0) -> None:
        points = standing_points(10_000)
        add_jump(points, takeoff=30, landing=45, height=0.15)
        self._points, self._delay = points, delay

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        time.sleep(self._delay)
        frame = round(timestamp_ms * FPS / 1000)
        return np.column_stack([self._points[frame], np.ones(len(self._points[frame]))])

    def close(self) -> None:
        pass


class _Analyzers:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    def build(self, models) -> MovementAnalyzer:
        extractor = VideoPoseExtractor(
            VideoFrameReader(), partial(_JumpEstimator, self._delay), max_duration_ms=round(MAX_SECONDS * 1000)
        )
        return MovementAnalyzer(extractor, LandmarkSeriesPreprocessor(0.1, 3), AngleCalculator())


class _World:
    def __init__(self, video: Path, delay: float = 0.0) -> None:
        self.video = video
        self.slots = AnalysisSlots(1)
        self.requested_hosts: list[str] = []
        settings = VideoDownloadSettings(ALLOWED_HOSTS=[HOST])
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._bucket))
        pipeline = MovementAnalysisPipeline(
            slots=self.slots,
            warmer=ModelWarmer(lambda: "models"),
            analyzers=_Analyzers(delay),
            strategies=MovementStrategyFactory(MovementPhaseSettings(), RiskSettings()).build_all(),
            source=HttpVideoSource(self.client, AllowedHostPolicy(settings.ALLOWED_HOSTS), settings),
            duration_guard=VideoDurationGuard(MAX_SECONDS),
            runner=ThreadedAnalysisRunner(progress_every_frames=30),
            sampler=FrameSampler(None),
        )
        servicer = MovementAnalysisServicer(pipeline, AnalysisMessages(RiskLevelClassifier(0.33, 0.66), "pose-v1", "risk-v1"))
        self.registration = partial(analysis_pb2_grpc.add_MovementAnalysisServiceServicer_to_server, servicer)

    def _bucket(self, request: httpx.Request) -> httpx.Response:
        self.requested_hosts.append(request.url.host)
        return httpx.Response(200, content=self.video.read_bytes())


def _request(url: str = VIDEO_URL, movement=movement_pb2.MOVEMENT_TYPE_JUMP) -> analysis_pb2.AnalyzeVideoRequest:
    return analysis_pb2.AnalyzeVideoRequest(video_url=url, movement_type=movement)


def _temporary_videos() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("kinesiapp-video-*"))


@pytest.fixture
def jump_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "jump.mp4", frame_count=JUMP_FRAMES, fps=FPS)


async def _failure(world: _World, request: analysis_pb2.AnalyzeVideoRequest) -> grpc.aio.AioRpcError:
    async with serving([world.registration]) as channel:
        stub = analysis_pb2_grpc.MovementAnalysisServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as raised:
            async for _ in stub.AnalyzeVideo(request, metadata=WITH_TOKEN):
                pass
    await world.client.aclose()
    return raised.value


def test_a_jump_streams_its_progress_and_ends_with_the_assessment(jump_video: Path) -> None:
    world = _World(jump_video)
    before = _temporary_videos()

    async def main():
        async with serving([world.registration]) as channel:
            stub = analysis_pb2_grpc.MovementAnalysisServiceStub(channel)
            responses = [response async for response in stub.AnalyzeVideo(_request(), metadata=WITH_TOKEN)]
        await world.client.aclose()
        return responses

    responses = asyncio.run(main())

    progress = [response.progress for response in responses[:-1]]
    assert [item.stage for item in progress[:3]] == [
        movement_pb2.ANALYSIS_STAGE_QUEUED,
        movement_pb2.ANALYSIS_STAGE_DOWNLOADING_VIDEO,
        movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE,
    ]
    extracting = [(item.frames_processed, item.frames_total) for item in progress if item.stage == movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE]
    assert extracting == [(0, 90), (30, 90), (60, 90), (90, 90)]
    assert [item.stage for item in progress[-2:]] == [
        movement_pb2.ANALYSIS_STAGE_DETECTING_REPETITIONS,
        movement_pb2.ANALYSIS_STAGE_SCORING_RISK,
    ]
    result = responses[-1].result
    assert [rep.detected_by for rep in result.risk_assessment.repetitions] == [movement_pb2.DETECTION_METHOD_LANDING]
    assert result.duration_ms == 3000 and result.angle_samples
    assert world.slots.in_use == 0 and _temporary_videos() <= before


def test_an_unspecified_movement_is_a_bad_request(jump_video: Path) -> None:
    error = asyncio.run(_failure(_World(jump_video), _request(movement=movement_pb2.MOVEMENT_TYPE_UNSPECIFIED)))

    assert (error.code(), error_reason(error)) == (grpc.StatusCode.INVALID_ARGUMENT, "BAD_REQUEST")


def test_a_url_outside_the_bucket_is_rejected_without_requesting_it(jump_video: Path) -> None:
    world = _World(jump_video)

    error = asyncio.run(_failure(world, _request(url="https://169.254.169.254/latest/meta-data")))

    assert (error.code(), error_reason(error)) == (grpc.StatusCode.INVALID_ARGUMENT, "BAD_REQUEST")
    assert world.requested_hosts == []


def test_a_video_longer_than_allowed_is_video_too_long(tmp_path: Path) -> None:
    long_video = write_video(tmp_path / "long.mp4", frame_count=int(MAX_SECONDS * FPS) + 1, fps=FPS)

    error = asyncio.run(_failure(_World(long_video), _request()))

    assert (error.code(), error_reason(error)) == (grpc.StatusCode.INVALID_ARGUMENT, "VIDEO_TOO_LONG")


def test_a_video_without_movement_is_no_movement_detected(jump_video: Path) -> None:
    world = _World(jump_video)

    error = asyncio.run(_failure(world, _request(movement=movement_pb2.MOVEMENT_TYPE_SQUAT)))

    assert (error.code(), error_reason(error)) == (grpc.StatusCode.FAILED_PRECONDITION, "NO_MOVEMENT_DETECTED")


def test_a_second_analysis_while_one_runs_is_resource_exhausted(jump_video: Path) -> None:
    world = _World(jump_video, delay=0.02)

    async def main():
        async with serving([world.registration]) as channel:
            stub = analysis_pb2_grpc.MovementAnalysisServiceStub(channel)
            first = stub.AnalyzeVideo(_request(), metadata=WITH_TOKEN)
            await first.read()
            with pytest.raises(grpc.aio.AioRpcError) as raised:
                async for _ in stub.AnalyzeVideo(_request(), metadata=WITH_TOKEN):
                    pass
            first.cancel()
        await world.client.aclose()
        return raised.value

    error = asyncio.run(main())

    assert (error.code(), error_reason(error)) == (grpc.StatusCode.RESOURCE_EXHAUSTED, "AI_BUSY")


def test_a_client_that_cancels_frees_the_slot_and_removes_the_video(jump_video: Path) -> None:
    world = _World(jump_video, delay=0.05)
    before = _temporary_videos()

    async def main():
        async with serving([world.registration]) as channel:
            stub = analysis_pb2_grpc.MovementAnalysisServiceStub(channel)
            call = stub.AnalyzeVideo(_request(), metadata=WITH_TOKEN)
            while (await call.read()).progress.stage != movement_pb2.ANALYSIS_STAGE_EXTRACTING_POSE:
                pass
            downloaded = _temporary_videos() - before
            call.cancel()
            for _ in range(100):
                if world.slots.in_use == 0 and not (_temporary_videos() - before):
                    break
                await asyncio.sleep(0.05)
        await world.client.aclose()
        return downloaded

    downloaded = asyncio.run(main())

    assert len(downloaded) == 1
    assert world.slots.in_use == 0
    assert not (_temporary_videos() - before)
