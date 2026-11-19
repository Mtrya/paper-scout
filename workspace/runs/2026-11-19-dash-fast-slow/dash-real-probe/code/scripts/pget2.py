#!/usr/bin/env python3
"""Parallel range downloader with journaling resume (stdlib only).

Usage: pget2.py URL OUT [NCONN]
Total size comes from a one-byte ranged GET (servers like ModelScope omit Content-Length on HEAD).
Journal: OUT.parts/<idx>.part + <idx>.done; re-run to resume.
"""
import concurrent.futures as cf
import os
import re
import sys
import urllib.request

CHUNK = 1 << 22


def probe(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
        total = int(re.search(r"/(\d+)$", cr).group(1))
        return total, r.geturl()


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
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            if os.path.getsize(path) >= want:
                break
        except Exception as e:  # noqa: BLE001
            print(f"retry {path} ({attempt}): {e!r}", flush=True)
    ok = os.path.exists(path) and os.path.getsize(path) >= end - start + 1
    if ok:
        open(done_path, "w").write("ok")
    else:
        raise RuntimeError(f"range {start}-{end} incomplete")
    return path


def main():
    url, out = sys.argv[1], sys.argv[2]
    nconn = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    if os.path.exists(out):
        print("exists:", out)
        return
    total, final_url = probe(url)
    print(f"{total / 1e9:.2f} GB from {final_url[:90]}", flush=True)
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    n = (total + CHUNK - 1) // CHUNK
    tasks = [(i * CHUNK, min(total - 1, (i + 1) * CHUNK - 1),
              os.path.join(parts_dir, f"{i:06d}.part"), os.path.join(parts_dir, f"{i:06d}.done")) for i in range(n)]
    done_ct = 0
    with cf.ThreadPoolExecutor(max_workers=nconn) as ex:
        futs = [ex.submit(fetch_range, final_url, *t) for t in tasks]
        for f in cf.as_completed(futs):
            f.result()
            done_ct += 1
            if done_ct % 100 == 0:
                print(f"{done_ct}/{n} ranges", flush=True)
    with open(out + ".tmp", "wb") as o:
        for i in range(n):
            with open(os.path.join(parts_dir, f"{i:06d}.part"), "rb") as p:
                while True:
                    b = p.read(1 << 22)
                    if not b:
                        break
                    o.write(b)
    assert os.path.getsize(out + ".tmp") == total, (os.path.getsize(out + ".tmp"), total)
    os.rename(out + ".tmp", out)
    for f in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, f))
    os.rmdir(parts_dir)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
