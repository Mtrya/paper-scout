#!/usr/bin/env python3
"""Resumable parallel HTTP range downloader with chunk journal.

Usage: pget2.py <url> <out> [threads]
- Preallocates <out>, downloads in fixed chunks with unlimited retries.
- Each finished chunk is journaled to <out>.done; restart skips journaled chunks.
"""
import os
import sys
import threading
import time

import requests


def main():
    url, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    r = requests.head(url, allow_redirects=True, timeout=30)
    final_url = r.url
    size = int(r.headers["Content-Length"])
    print(f"size {size/1e9:.2f}GB from {final_url[:80]}", flush=True)
    done_file = out + ".done"
    done_chunks = set()
    if os.path.exists(done_file):
        done_chunks = {int(x) for x in open(done_file).read().split() if x.strip()}
    if not os.path.exists(out) or os.path.getsize(out) != size:
        with open(out, "wb") as f:
            f.truncate(size)
        done_chunks = set()
        open(done_file, "w").close()
    chunk = 256 * (1 << 20)  # 256MB chunks: small rework per failure, fine journal
    todo = [i for i in range((size + chunk - 1) // chunk) if i not in done_chunks]
    print(f"{len(done_chunks)} chunks already done, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    done_bytes = [len(done_chunks) * chunk]
    queue = list(todo)
    queue_lock = threading.Lock()

    def worker():
        while True:
            with queue_lock:
                if not queue:
                    return
                i = queue.pop(0)
            start = i * chunk
            end = min(size, start + chunk) - 1
            if start > end:
                continue
            attempt = 0
            while True:
                try:
                    h = {"Range": f"bytes={start}-{end}"}
                    with requests.get(final_url, headers=h, stream=True, timeout=300) as resp:
                        if resp.status_code != 206:
                            raise RuntimeError(f"expected 206, got {resp.status_code}")
                        buf = []
                        n = 0
                        for data in resp.iter_content(1 << 20):
                            buf.append(data)
                            n += len(data)
                            with lock:
                                done_bytes[0] += len(data)
                        if n != end - start + 1:
                            raise RuntimeError(f"short read {n} != {end-start+1}")
                        with open(out, "r+b") as f:
                            f.seek(start)
                            f.write(b"".join(buf))
                    with lock:
                        with open(done_file, "a") as df:
                            df.write(f"{i}\n")
                    break
                except Exception as e:
                    attempt += 1
                    wait = min(60, 2 ** min(attempt, 6))
                    print(f"chunk {i} attempt {attempt} failed: {e}; retry in {wait}s", flush=True)
                    time.sleep(wait)

    ths = [threading.Thread(target=worker) for _ in range(threads)]
    t0 = time.time()
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(15)
        el = time.time() - t0
        print(f"{done_bytes[0]/1e9:.2f}GB / {size/1e9:.2f}GB  {done_bytes[0]/el/1e6:.1f}MB/s", flush=True)
    for t in ths:
        t.join()
    assert os.path.getsize(out) == size
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
