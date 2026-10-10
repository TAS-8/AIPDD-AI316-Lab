# data_ingestion.py
import numpy as np
from typing import Optional

class DataIngestion:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.is_connected = False

    def connect(self) -> bool:
        """Establishes a connection to the RTSP stream."""
        pass

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads and returns the next raw frame from the camera."""
        pass
        
    def release(self) -> None:
        """Closes the stream and releases resources."""
        pass
