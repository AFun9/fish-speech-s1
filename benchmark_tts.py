"""TTS API speed benchmark — supports both blocking and streaming modes."""

import argparse
import time

import requests

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit PCM


def bytes_to_duration(n_bytes: int, header: int = 44) -> float:
    return max(0, n_bytes - header) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


def run_blocking(url: str, payload: dict) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    audio_dur = bytes_to_duration(len(resp.content))
    return {"latency": elapsed, "audio_dur": audio_dur, "size": len(resp.content)}


def run_streaming(url: str, payload: dict) -> dict:
    payload = {**payload, "streaming": True, "format": "wav"}
    t0 = time.perf_counter()
    ttfc = None
    total_bytes = 0
    chunk_count = 0

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if ttfc is None:
                ttfc = time.perf_counter() - t0
            total_bytes += len(chunk)
            chunk_count += 1

    elapsed = time.perf_counter() - t0
    audio_dur = bytes_to_duration(total_bytes)
    return {
        "latency": elapsed,
        "ttfc": ttfc or elapsed,
        "audio_dur": audio_dur,
        "size": total_bytes,
        "chunks": chunk_count,
    }


def fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


def benchmark(url: str, text: str, num_warmup: int, num_runs: int, streaming: bool):
    mode = "streaming" if streaming else "blocking"
    runner = run_streaming if streaming else run_blocking

    payload = {
        "text": text,
        "references": [],
        "reference_id": None,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "format": "wav",
    }

    print(f"Mode: {mode} | Warmup: {num_warmup} | Runs: {num_runs}")
    print(f"Text: {text!r}\n")

    for i in range(num_warmup):
        r = runner(url, payload)
        tag = f"[Warmup {i + 1}/{num_warmup}]"
        if streaming:
            print(f"{tag}  TTFC={fmt(r['ttfc'])}  total={fmt(r['latency'])}  audio={r['audio_dur']:.2f}s  chunks={r['chunks']}")
        else:
            print(f"{tag}  latency={fmt(r['latency'])}  audio={r['audio_dur']:.2f}s")

    results = []
    for i in range(num_runs):
        r = runner(url, payload)
        results.append(r)
        tag = f"[Run {i + 1}/{num_runs}]"
        if streaming:
            print(f"{tag}  TTFC={fmt(r['ttfc'])}  total={fmt(r['latency'])}  audio={r['audio_dur']:.2f}s  chunks={r['chunks']}")
        else:
            print(f"{tag}  latency={fmt(r['latency'])}  audio={r['audio_dur']:.2f}s")

    # Summary
    avg = lambda key: sum(r[key] for r in results) / len(results)
    mn = lambda key: min(r[key] for r in results)
    mx = lambda key: max(r[key] for r in results)

    avg_lat = avg("latency")
    avg_audio = avg("audio_dur")
    rtf = avg_lat / avg_audio if avg_audio > 0 else float("inf")

    print("\n" + "=" * 60)
    print(f"  Mode:            {mode}")
    print(f"  Avg latency:     {fmt(avg_lat)}  (min {fmt(mn('latency'))}, max {fmt(mx('latency'))})")
    if streaming:
        print(f"  Avg TTFC:        {fmt(avg('ttfc'))}  (min {fmt(mn('ttfc'))}, max {fmt(mx('ttfc'))})")
    print(f"  Avg audio len:   {avg_audio:.2f}s")
    print(f"  Avg RTF:         {rtf:.3f}x  (< 1.0 = faster than real-time)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS API benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--text", default="我能说中文了，这是一个速度测试。")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    benchmark(args.url, args.text, args.warmup, args.runs, args.streaming)
