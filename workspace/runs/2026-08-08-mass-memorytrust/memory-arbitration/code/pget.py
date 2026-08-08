#!/usr/bin/env python3
"""Parallel HTTP range downloader: pget.py <url> <out> [threads]"""
import os, sys, threading, requests

def main():
    url, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    # resolve redirects, get size (fall back to range probe if HEAD lacks length)
    r = requests.head(url, allow_redirects=True, timeout=30)
    size = r.headers.get('Content-Length')
    if size is None:
        rg = requests.get(url, headers={'Range': 'bytes=0-0'}, allow_redirects=True, timeout=30)
        assert rg.status_code == 206, f"range probe got {rg.status_code}"
        size = rg.headers['Content-Range'].split('/')[-1]
        final_url = rg.url
    else:
        final_url = r.url
    size = int(size)
    print(f"size={size/1e9:.2f}GB threads={threads}", flush=True)
    # preallocate
    with open(out, 'wb') as f:
        f.truncate(size)
    chunk = (size + threads - 1) // threads
    lock = threading.Lock()
    done = [0]
    def worker(i):
        start = i * chunk
        end = min(size, start + chunk) - 1
        if start > end:
            return
        for attempt in range(5):
            try:
                h = {'Range': f'bytes={start}-{end}'}
                with requests.get(final_url, headers=h, stream=True, timeout=120) as resp:
                    if resp.status_code != 206:
                        raise RuntimeError(f"expected 206, got {resp.status_code}")
                    with open(out, 'r+b') as f:
                        f.seek(start)
                        for data in resp.iter_content(1 << 20):
                            f.write(data)
                            with lock:
                                done[0] += len(data)
                return
            except Exception as e:
                print(f"worker {i} attempt {attempt} failed: {e}", flush=True)
        raise RuntimeError(f"worker {i} gave up")
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    import time
    t0 = time.time()
    for t in ths: t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(15)
        el = time.time() - t0
        print(f"{done[0]/1e9:.2f}GB / {size/1e9:.2f}GB  {done[0]/el/1e6:.1f}MB/s", flush=True)
    for t in ths: t.join()
    assert os.path.getsize(out) == size
    print("DONE", out, flush=True)

if __name__ == '__main__':
    main()
