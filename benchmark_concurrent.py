"""Concurrent TTS benchmark — measures throughput and latency at various concurrency levels."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2


def single_request(url: str, text: str, streaming: bool):
    payload = {
        "text": text,
        "references": [],
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "format": "wav",
        "streaming": streaming,
    }

    t0 = time.perf_counter()
    ttfa = None  # time to first audio (first PCM data after 44-byte WAV header)
    total_bytes = 0

    if streaming:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=4096):
                total_bytes += len(chunk)
                if ttfa is None and total_bytes > 44:
                    ttfa = time.perf_counter() - t0
    else:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        total_bytes = len(resp.content)
        ttfa = time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    audio_dur = max(0, total_bytes - 44) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    return {
        "latency": elapsed,
        "ttfa": ttfa or elapsed,
        "audio_dur": audio_dur,
    }


def fmt(s):
    return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


def run_concurrency_level(url, text, concurrency, num_requests, streaming):
    results = []
    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(single_request, url, text, streaming)
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"latency": 0, "ttfc": 0, "audio_dur": 0, "error": str(e)})

    wall_time = time.perf_counter() - t_wall_start

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        print(f"  Concurrency={concurrency}: ALL FAILED ({len(errors)} errors)")
        return

    avg_lat = sum(r["latency"] for r in ok) / len(ok)
    avg_ttfa = sum(r["ttfa"] for r in ok) / len(ok)
    min_ttfa = min(r["ttfa"] for r in ok)
    max_ttfa = max(r["ttfa"] for r in ok)
    total_audio = sum(r["audio_dur"] for r in ok)
    throughput = total_audio / wall_time

    print(f"  C={concurrency:>2}  "
          f"reqs={len(ok)}/{num_requests}  "
          f"avg_lat={fmt(avg_lat)}  "
          f"TTFA avg={fmt(avg_ttfa)} min={fmt(min_ttfa)} max={fmt(max_ttfa)}  "
          f"throughput={throughput:.1f}x  "
          f"wall={fmt(wall_time)}")


def main():
    parser = argparse.ArgumentParser(description="Concurrent TTS Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--text", default="在人工智能领域，语音合成技术已经取得了巨大的进步。")
    parser.add_argument("--requests", type=int, default=10,
                        help="Total requests per concurrency level")
    parser.add_argument("--levels", type=str, default="1,2,4,6,8,10",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before benchmarking")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    mode = "streaming" if args.streaming else "blocking"

    print(f"URL: {args.url}")
    print(f"Text: {args.text!r}")
    print(f"Mode: {mode}")
    print(f"Requests per level: {args.requests}")
    print(f"Concurrency levels: {levels}")
    print()

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} requests)...")
        for _ in range(args.warmup):
            try:
                single_request(args.url, args.text, args.streaming)
            except Exception as e:
                print(f"  Warmup error: {e}")
        print()

    print("Benchmarking:")
    for level in levels:
        run_concurrency_level(args.url, args.text, level, args.requests, args.streaming)

    print("\nDone!")
    print("  throughput = total audio seconds / wall clock seconds")
    print("  Higher throughput = better GPU utilization")


if __name__ == "__main__":
    main()
