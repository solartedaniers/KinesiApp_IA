import asyncio
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from kinesiapp_ai.analysis.errors import AnalysisCancelledError
from kinesiapp_ai.analysis.movement_analyzer import AnalysisStage, MovementAnalysis, MovementAnalyzer
from kinesiapp_ai.analysis.camera_view import CameraView
from kinesiapp_ai.analysis.movement_strategy import MovementStrategy


@dataclass(frozen=True)
class StageReached:
    stage: AnalysisStage


@dataclass(frozen=True)
class FramesProcessed:
    processed: int


AnalysisEvent = StageReached | FramesProcessed

_FINISHED = object()


class _ForwardingObserver:
    """Paso el avance del hilo al event loop y corto el análisis en cuanto el cliente cancela."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, cancelled: threading.Event, every_frames: int
    ) -> None:
        self._loop = loop
        self._queue = queue
        self._cancelled = cancelled
        self._every_frames = every_frames

    def on_stage(self, stage: AnalysisStage) -> None:
        self._send(StageReached(stage))

    def on_frames(self, processed: int) -> None:
        if processed % self._every_frames == 0:
            self._send(FramesProcessed(processed))
        elif self._cancelled.is_set():
            raise AnalysisCancelledError

    def _send(self, event: AnalysisEvent) -> None:
        if self._cancelled.is_set():
            raise AnalysisCancelledError
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


class ThreadedAnalysisRunner:
    """Corro el analizador en un hilo aparte y entrego su avance como un stream; el último elemento es el resultado.

    Lo saco del event loop porque YOLO y MediaPipe bloquean decenas de segundos: si no, el servidor dejaría
    de responder la salud, WarmUp y los demás streams mientras dura el análisis.
    """

    def __init__(self, progress_every_frames: int) -> None:
        self._progress_every_frames = progress_every_frames

    async def run(
        self, analyzer: MovementAnalyzer, video_path: Path, strategies: Mapping[CameraView, MovementStrategy]
    ) -> AsyncIterator[AnalysisEvent | MovementAnalysis]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        observer = _ForwardingObserver(loop, queue, cancelled, self._progress_every_frames)
        work = asyncio.ensure_future(asyncio.to_thread(analyzer.analyze, video_path, strategies, observer))
        work.add_done_callback(lambda _: queue.put_nowait(_FINISHED))
        try:
            while (event := await queue.get()) is not _FINISHED:
                yield event
            yield work.result()
        finally:
            cancelled.set()
            if not work.done():
                # Espero a que el hilo suelte el video: así el archivo se puede borrar y el cupo no se libera
                # mientras el análisis anterior todavía ocupa memoria
                with suppress(Exception):
                    await asyncio.shield(work)
