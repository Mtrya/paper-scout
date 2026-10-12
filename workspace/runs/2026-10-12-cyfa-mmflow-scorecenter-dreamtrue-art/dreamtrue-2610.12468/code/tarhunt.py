"""HTTP-range random access helpers for large remote tar files."""
import urllib.request, re

def fetch_range(url, off, span):
    req = urllib.request.Request(url, headers={'Range': f'bytes={off}-{off+span-1}'})
    return urllib.request.urlopen(req, timeout=60).read()

def scan_headers(buf, base_off=0):
    """Find tar headers in buf; yield (abs_off, name, size)."""
    i = buf.find(b'ustar')
    while i != -1:
        hs = i - 257
        if hs >= 0:
            hdr = buf[hs:hs+512]
            name = hdr[:100].split(b'\0')[0].decode(errors='replace')
            try:
                size = int(hdr[124:136].split(b'\0')[0].strip() or b'0', 8)
            except ValueError:
                size = -1
            yield base_off + hs, name, size
        i = buf.find(b'ustar', i + 1)

def scan_region(url, start, end, chunk=4*1024*1024, callback=None):
    """Scan [start,end) in chunks; collect entry headers matching {task}/{ep}/."""
    found = []
    off = start // 512 * 512
    while off < end:
        buf = fetch_range(url, off, chunk)
        for hoff, name, size in scan_headers(buf, off):
            m = re.match(r'(\d+)/(\d+)/', name)
            if m:
                found.append((hoff, int(m.group(1)), int(m.group(2)), name))
        if callback: callback(off, len(found))
        off += chunk
    return found

def read_entry(url, hoff, size):
    """Read the file content for a header at hoff."""
    data_off = (hoff // 512 + 1) * 512
    return fetch_range(url, data_off, size)
