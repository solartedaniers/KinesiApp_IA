class VideoAnalysisError(Exception):
    """Base for expected failures of a video analysis (as opposed to bugs)."""


class UnreadableVideoError(VideoAnalysisError):
    pass


class NoMovementDetectedError(VideoAnalysisError):
    pass


class VideoTooLongError(VideoAnalysisError):
    pass


class AnalysisCancelledError(Exception):
    """La lanzo dentro del hilo del análisis cuando el cliente cancela, para soltar el video y la memoria cuanto antes."""
