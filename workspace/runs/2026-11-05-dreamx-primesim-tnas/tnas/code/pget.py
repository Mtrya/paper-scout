#!/usr/bin/env python3
"""Parallel range downloader with journaling resume (stdlib only).

Usage: pget.py URL OUT [NCONN]
Journal: OUT.parts/<idx>.part files; done markers in OUT.parts/<idx>.done.
Re-run to resume. On success assembles OUT and removes OUT.parts.
"""
import concurrent.futures as cf
import os
import sys
import urllib.request

CHUNK = 1 << 22  # 4MB per range request


def fetch_range(url, start, end, path, done_path, retries=10):
    if os.path.exists(done_path):
        return path
    for attempt in range(retries):
        try:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            want = end - start + 1
            if have >= want:
                break
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            if os.path.getsize(path) >= want:
                break
        except Exception as e:
            print(f"retry {path} ({attempt}): {e!r}", flush=True)
    ok = os.path.exists(path) and os.path.getsize(path) >= end - start + 1
    if ok:
        open(done_path, "w").write("ok")
    else:
        raise RuntimeError(f"range {start}-{end} incomplete")
    return path


def main():
    url, out = sys.argv[1], sys.argv[2]
    nconn = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    if os.path.exists(out):
        print("exists:", out)
        return
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Length"])
        final_url = r.geturl()
    print(f"{total / 1e9:.2f} GB from {final_url[:80]}", flush=True)
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    n = (total + CHUNK - 1) // CHUNK
    tasks = []
    for i in range(n):
        s, e = i * CHUNK, min(total - 1, (i + 1) * CHUNK - 1)
        tasks.append((s, e, os.path.join(parts_dir, f"{i:06d}.part"), os.path.join(parts_dir, f"{i:06d}.done")))
    done_ct = 0
    with cf.ThreadPoolExecutor(max_workers=nconn) as ex:
        futs = [ex.submit(fetch_range, final_url, *t) for t in tasks]
        for f in cf.as_completed(futs):
            f.result()
            done_ct += 1
            if done_ct % 50 == 0:
                print(f"{done_ct}/{n} ranges", flush=True)
    with open(out + ".tmp", "wb") as o:
        for i in range(n):
            with open(os.path.join(parts_dir, f"{i:06d}.part"), "rb") as p:
                while True:
                    b = p.read(1 << 22)
                    if not b:
                        break
                    o.write(b)
    assert os.path.getsize(out + ".tmp") == total
    os.rename(out + ".tmp", out)
    for f in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, f))
    os.rmdir(parts_dir)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
