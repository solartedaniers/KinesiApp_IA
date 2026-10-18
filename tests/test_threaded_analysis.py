"""Pruebo el puente entre el hilo del análisis y el event loop: orden, errores y cancelación."""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kinesiapp_ai.analysis.errors import NoMovementDetectedError
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage
from kinesiapp_ai.pipeline.threaded_analysis import FramesProcessed, StageReached, ThreadedAnalysisRunner

RESULT = object()


class _ScriptedAnalyzer:
    """Simulo un análisis de `frames` frames en un hilo; opcionalmente falla al final."""

    def __init__(self, frames: int, delay: float = 0.0, error: Exception | None = None) -> None:
        self.frames, self.delay, self.error = frames, delay, error
        self.finished = threading.Event()
        self.thread_name = ""

    def analyze(self, video_path: Path, strategy, observer):
        self.thread_name = threading.current_thread().name
        try:
            observer.on_stage(AnalysisStage.EXTRACTING_POSE)
            for processed in range(1, self.frames + 1):
                time.sleep(self.delay)
                observer.on_frames(processed)
            observer.on_stage(AnalysisStage.DETECTING_REPETITIONS)
            if self.error:
                raise self.error
            observer.on_stage(AnalysisStage.SCORING_RISK)
            return RESULT
        finally:
            self.finished.set()


async def _collect(analyzer, every: int = 2) -> list:
    return [event async for event in ThreadedAnalysisRunner(every).run(analyzer, Path("v.mp4"), strategy=None)]


def test_events_arrive_in_order_throttled_and_the_result_comes_last() -> None:
    analyzer = _ScriptedAnalyzer(frames=5)

    events = asyncio.run(_collect(analyzer))

    assert events == [
        StageReached(AnalysisStage.EXTRACTING_POSE),
        FramesProcessed(2),
        FramesProcessed(4),
        StageReached(AnalysisStage.DETECTING_REPETITIONS),
        StageReached(AnalysisStage.SCORING_RISK),
        RESULT,
    ]
    assert analyzer.thread_name != threading.main_thread().name


def test_an_analysis_error_reaches_the_stream_after_its_events() -> None:
    received = []

    async def main():
        async for event in ThreadedAnalysisRunner(10).run(
            _ScriptedAnalyzer(frames=3, error=NoMovementDetectedError("nothing")), Path("v.mp4"), strategy=None
        ):
            received.append(event)

    with pytest.raises(NoMovementDetectedError):
        asyncio.run(main())

    assert received[-1] == StageReached(AnalysisStage.DETECTING_REPETITIONS)


def test_the_event_loop_keeps_answering_while_the_analysis_runs() -> None:
    async def main():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beating = asyncio.create_task(heartbeat())
        await _collect(_ScriptedAnalyzer(frames=20, delay=0.01))
        beating.cancel()
        return ticks

    assert asyncio.run(main()) >= 5


def test_cancelling_stops_the_thread_and_waits_for_it_before_returning() -> None:
    analyzer = _ScriptedAnalyzer(frames=10_000, delay=0.001)

    async def main():
        consumer = asyncio.create_task(_collect(analyzer, every=1))
        await asyncio.sleep(0.1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        return analyzer.finished.is_set()

    started = time.monotonic()
    finished_when_cancelled = asyncio.run(main())

    assert finished_when_cancelled
    # Si el hilo no se detuviera, recorrer los 10 000 frames tardaría más de 10 s
    assert time.monotonic() - started < 3
