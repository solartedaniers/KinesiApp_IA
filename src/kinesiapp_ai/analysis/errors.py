class VideoAnalysisError(Exception):
    """Base for expected failures of a video analysis (as opposed to bugs)."""


class UnreadableVideoError(VideoAnalysisError):
    pass


class NoMovementDetectedError(VideoAnalysisError):
    pass
