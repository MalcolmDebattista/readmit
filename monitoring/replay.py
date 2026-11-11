"""Replay real holdout discharges against POST /risk at a fixed rate and report latency percentiles and throughput.
Standard library only, so it runs with any Python 3.

    python3 monitoring/replay.py --rate 5 --count 1500
    python3 monitoring/replay.py --rate 20 --count 1200 --label burst
"""
import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8888/risk")
    ap.add_argument("--file", default="data/processed/holdout_requests.jsonl")
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second (open loop)")
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--label", default="steady")
    ap.add_argument("--out", default=None, help="append the summary (JSON) to this file")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds before a request counts as an error")
    args = ap.parse_args()

    bodies = [line for line in open(args.file, encoding="utf-8")][: args.count]
    lat, errors, lock = [], [0], threading.Lock()

    def send(body):
        req = urllib.request.Request(args.url, data=body.encode(), headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                r.read()
            with lock:
                lat.append(time.perf_counter() - t0)
        except (urllib.error.URLError, TimeoutError):
            with lock:
                errors[0] += 1

    start, threads = time.perf_counter(), []
    for i, body in enumerate(bodies):                     # open loop: send on schedule, whatever the replies do
        delay = start + i / args.rate - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=send, args=(body,)); th.start(); threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - start

    ms = sorted(x * 1000 for x in lat) or [float("nan")]   # nan when nothing came back in time
    q = lambda p: ms[min(len(ms) - 1, int(round(p * (len(ms) - 1))))]  # noqa: E731
    summary = {"label": args.label, "rate_target": args.rate, "sent": len(bodies), "ok": len(lat), "errors": errors[0],
               "seconds": round(wall, 1), "throughput_per_s": round(len(lat) / wall, 2),
               "p50_ms": round(q(0.50), 1), "p95_ms": round(q(0.95), 1), "p99_ms": round(q(0.99), 1),
               "max_ms": round(ms[-1], 1), "mean_ms": round(statistics.mean(ms), 1), "timeout_s": args.timeout,
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
