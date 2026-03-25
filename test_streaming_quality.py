"""Compare streaming vs non-streaming audio quality and timing."""

import time

import requests

API_URL = "http://127.0.0.1:8080/v1/tts"
TEXT = "在人工智能领域，语音合成技术已经取得了巨大的进步。现代的TTS系统能够生成非常自然和流畅的语音，几乎与真人无法区分。"


def fetch_streaming(path: str):
    payload = {
        "text": TEXT,
        "references": [],
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "format": "wav",
        "streaming": True,
        "seed": 42,
    }
    t0 = time.perf_counter()
    ttfc = None
    chunks = []
    chunk_times = []

    with requests.post(API_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            now = time.perf_counter() - t0
            if ttfc is None:
                ttfc = now
            chunks.append(chunk)
            chunk_times.append(now)

    total = time.perf_counter() - t0
    data = b"".join(chunks)

    with open(path, "wb") as f:
        f.write(data)

    # Detect chunk arrival groups (gaps > 50ms indicate separate DAC segments)
    segments = 1
    for i in range(1, len(chunk_times)):
        if (chunk_times[i] - chunk_times[i - 1]) > 0.05:
            segments += 1

    audio_dur = max(0, len(data) - 44) / (24000 * 2)
    print(f"  [Streaming]  TTFC={ttfc*1000:.0f}ms  total={total*1000:.0f}ms  "
          f"audio={audio_dur:.2f}s  segments={segments}  saved={path}")
    return audio_dur


def fetch_blocking(path: str):
    payload = {
        "text": TEXT,
        "references": [],
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "format": "wav",
        "streaming": False,
        "seed": 42,
    }
    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    total = time.perf_counter() - t0

    with open(path, "wb") as f:
        f.write(resp.content)

    audio_dur = max(0, len(resp.content) - 44) / (24000 * 2)
    print(f"  [Blocking]   latency={total*1000:.0f}ms  "
          f"audio={audio_dur:.2f}s  saved={path}")
    return audio_dur


if __name__ == "__main__":
    print("Generating audio (same text, same seed)...\n")

    for i in range(2):
        print(f"--- Round {i + 1} ---")
        fetch_streaming(f"test_streaming_{i}.wav")
        fetch_blocking(f"test_blocking_{i}.wav")
        print()

    print("Done! Compare the audio files:")
    print("  test_streaming_*.wav  — decoded in chunks (streaming)")
    print("  test_blocking_*.wav   — decoded all at once (baseline)")
    print("\nListen for clicks, pops, or discontinuities at segment boundaries.")
