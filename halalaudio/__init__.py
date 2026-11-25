"""Flag music in a video or audio soundtrack, so creators can keep their audio halal."""
from .analyse import Report, Segment, Window, analyse, check_file, load_audio, timestamp

__version__ = "1.0.0"
__all__ = ["analyse", "check_file", "load_audio", "Report", "Segment", "Window", "timestamp"]
