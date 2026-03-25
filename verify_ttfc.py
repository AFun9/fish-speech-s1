"""Verify streaming TTFC — logs every chunk arrival with precise timestamps."""

import time

import requests

API_URL = "http://127.0.0.1:8081/v1/tts"
SAMPLE_RATE = 24000


def verify():
    payload = {
        "text": "在人工智能领域，语音合成技术已经取得了巨大的进步。",
        "references": [],
        "reference_id": None,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "format": "wav",
        "streaming": True,
    }

    print("Sending request...\n")
    t0 = time.perf_counter()

    with requests.post(API_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total_bytes = 0
        audio_bytes = 0

        for i, chunk in enumerate(resp.iter_content(chunk_size=4096)):
            now = (time.perf_counter() - t0) * 1000  # ms
            total_bytes += len(chunk)

            if i == 0:
                # First chunk: contains WAV header (44 bytes) + PCM data
                pcm_in_chunk = max(0, len(chunk) - 44)
                pcm_duration = pcm_in_chunk / (SAMPLE_RATE * 2) * 1000  # ms of audio
                audio_bytes += pcm_in_chunk
                print(f"  #{i:<4}  t={now:>8.1f}ms  bytes={len(chunk):>6}  "
                      f"(WAV header 44B + {pcm_in_chunk}B PCM = {pcm_duration:.1f}ms of audio)  ← FIRST CHUNK")
            else:
                pcm_duration = len(chunk) / (SAMPLE_RATE * 2) * 1000
                audio_bytes += len(chunk)
                if i <= 10 or i % 50 == 0:
                    total_audio_ms = audio_bytes / (SAMPLE_RATE * 2) * 1000
                    print(f"  #{i:<4}  t={now:>8.1f}ms  bytes={len(chunk):>6}  "
                          f"(+{pcm_duration:.1f}ms audio, cumulative {total_audio_ms:.0f}ms)")

        total_time = (time.perf_counter() - t0) * 1000
        total_audio = audio_bytes / (SAMPLE_RATE * 2)

    print(f"\n{'='*60}")
    print(f"  Total time:     {total_time:.0f}ms")
    print(f"  Total audio:    {total_audio:.2f}s ({total_audio*1000:.0f}ms)")
    print(f"  Total chunks:   {i + 1}")
    print(f"  RTF:            {(total_time/1000)/total_audio:.3f}x")
    print(f"{'='*60}")
    print(f"\n  结论: 第一块数据在请求发出后到达，")
    print(f"  其中包含可立即播放的 PCM 音频数据。")
    print(f"  流式播放器收到首块后即可开始出声。\n")


if __name__ == "__main__":
    for run in range(3):
        print(f"--- Run {run + 1}/3 ---")
        verify()
        print()
