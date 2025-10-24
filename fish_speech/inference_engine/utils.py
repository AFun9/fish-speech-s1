import io
import wave
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    code: Literal["header", "segment", "error", "final"]
    audio: Optional[Tuple[int, np.ndarray]]
    error: Optional[Exception]


def wav_chunk_header(
    sample_rate: int = 44100, bit_depth: int = 16, channels: int = 1
) -> bytes:
    """
    Create a WAV header for streaming audio.
    Uses maximum size values (0xFFFFFFFF - 8 for RIFF, 0xFFFFFFFF for data)
    to indicate unknown length, which is valid for streaming.
    """
    import struct
    
    byte_rate = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    
    # Create WAV header with maximum size for streaming
    # RIFF header
    riff_header = b'RIFF'
    # File size - 8 (use max value for streaming: 0xFFFFFFFF - 8)
    file_size = struct.pack('<I', 0xFFFFFFF7)  # 0xFFFFFFFF - 8
    wave_header = b'WAVE'
    
    # fmt subchunk
    fmt_header = b'fmt '
    fmt_chunk_size = struct.pack('<I', 16)  # PCM format chunk size
    audio_format = struct.pack('<H', 1)  # PCM = 1
    num_channels = struct.pack('<H', channels)
    sample_rate_bytes = struct.pack('<I', sample_rate)
    byte_rate_bytes = struct.pack('<I', byte_rate)
    block_align_bytes = struct.pack('<H', block_align)
    bits_per_sample = struct.pack('<H', bit_depth)
    
    # data subchunk header
    data_header = b'data'
    # Data size (use max value for streaming)
    data_size = struct.pack('<I', 0xFFFFFFFF)
    
    # Combine all parts
    wav_header = (
        riff_header + file_size + wave_header +
        fmt_header + fmt_chunk_size + audio_format + num_channels +
        sample_rate_bytes + byte_rate_bytes + block_align_bytes +
        bits_per_sample + data_header + data_size
    )
    
    return wav_header
