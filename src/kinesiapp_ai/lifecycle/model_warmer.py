import asyncio
import enum
import logging
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

Models = TypeVar("Models")


class WarmState(str, enum.Enum):
    WARMING_UP = "warming_up"
    READY = "ready"


class ModelWarmer(Generic[Models]):
    """Cargo los modelos una sola vez, en un hilo aparte, y dejo que cualquiera espere a que estén listos.

    Lo hago en un hilo porque cargar MediaPipe y YOLO bloquea varios segundos: en el event loop dejaría sin
    responder a todas las demás llamadas, también a la salud.
    """

    def __init__(self, load: Callable[[], Models]) -> None:
        self._load = load
        self._loading: asyncio.Task[Models] | None = None

    def warm_up(self) -> WarmState:
        """Empiezo la carga si no está en curso y respondo sin esperarla."""
        task = self._start()
        return WarmState.READY if task.done() and not _failed(task) else WarmState.WARMING_UP

    async def models(self) -> Models:
        return await asyncio.shield(self._start())

    def _start(self) -> asyncio.Task[Models]:
        # Si la carga anterior falló, la reintento en la próxima llamada en vez de quedar rota para siempre
        if self._loading is None or (self._loading.done() and _failed(self._loading)):
            self._loading = asyncio.create_task(asyncio.to_thread(self._load))
            self._loading.add_done_callback(_log_outcome)
        return self._loading


def _failed(task: asyncio.Task) -> bool:
    return task.cancelled() or task.exception() is not None


def _log_outcome(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("Model loading was cancelled")
    elif task.exception() is not None:
        logger.error("Model loading failed", exc_info=task.exception())
    else:
        logger.info("Models loaded")
