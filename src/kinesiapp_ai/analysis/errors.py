class VideoAnalysisError(Exception):
    """Base de las fallas esperadas de un análisis de video, para distinguirlas de los errores de código."""


class UnreadableVideoError(VideoAnalysisError):
    pass


class NoMovementDetectedError(VideoAnalysisError):
    pass


class VideoTooLongError(VideoAnalysisError):
    pass


class AnalysisCancelledError(Exception):
    """La lanzo dentro del hilo del análisis cuando el cliente cancela, para soltar el video y la memoria cuanto antes."""
