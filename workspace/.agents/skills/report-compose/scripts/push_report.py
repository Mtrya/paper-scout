#!/usr/bin/env python3
"""Paper Scout 报告飞书推送:分段推送 + 锚点插图 + 验证 + IM 通知。

把一次巡航的 `runs/<run-id>/report.docxxml` 发布为飞书文档:
1. 按顶层块边界(每元素一行)切成 ≤5.8KB 的段,首段创建文档,其余
   `docs +update --command append` 逐段追加(长文档单次写入会被静默截断)。
2. 若存在 `runs/<run-id>/assets/figures.json`,按锚点插图:
   锚点 `[[figure-anchor:<name>]]` 独占段落在导入时会被飞书丢弃,因此插图采用
   块位置法——从源文件取锚点的前一个段落文本,在带块 id 的抓取结果里定位该段落,
   `docs +media-insert` 插图到末尾,再 `block_move_after` 移到位;
   同一锚点多张图按清单顺序链式移动。
3. 重新抓取核对 img 数量。
4. 可选:给用户发 IM 私信(--user-id,用 --text 让 URL 展开成文档卡片)。

figures.json 格式(键为锚点名,与 report.docxxml 中的 [[figure-anchor:<name>]] 对应):
{
  "vtla:teaser": {"file": "assets/vtla-teaser.jpg", "caption": "图注"},
  "vipe:strips": [
    {"file": "assets/a.jpg", "caption": "第一张"},
    {"file": "assets/b.jpg", "caption": "第二张(链在前一张之后)"}
  ]
}
file 相对运行包目录。锚点名必须是顶层独立段落。

用法(在 workspace/ 下运行):
  python .agents/skills/report-compose/scripts/push_report.py <run-id> [--dry-run]
  python .agents/skills/report-compose/scripts/push_report.py <run-id> \\
      --user-id ou_xxx [--im-text "可选自定义私信文本"]

注意:每步都检查 lark-cli 返回 JSON 的 ok 字段;错误可能只走 stderr,
不能凭 "Command executed successfully" 字样判断成功。
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

CHUNK_LIMIT = 5800  # 字节;单次 --content 超过 ~10KB 会被静默截断,留足余量
DEFAULT_CLI = "npx --yes @larksuite/cli"


def run_cli(cli: str, args: list[str], cwd: Path) -> dict:
    """Run lark-cli, return parsed JSON. Raises on failure with the error message."""
    cmd = shlex.split(cli) + args
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    out = "\n".join(
        l for l in (proc.stdout + "\n" + proc.stderr).splitlines()
        if not re.search(r"cookie|inspiresession", l, re.I)
    )
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        raise RuntimeError(f"lark-cli 无 JSON 输出: {' '.join(args[:3])}\n{out[:500]}")
    d = json.loads(m.group(0))
    if not d.get("ok"):
        err = d.get("error", {})
        raise RuntimeError(f"lark-cli 失败: {' '.join(args[:3])}: {err.get('message', d)}")
    return d


def chunk_docxxml(text: str) -> list[str]:
    """按非空行(每个顶层块一行)切 ≤CHUNK_LIMIT 字节的段,不跨块。"""
    chunks, cur = [], ""
    for line in (l for l in text.split("\n") if l.strip()):
        if cur and len((cur + line).encode()) > CHUNK_LIMIT:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    if not chunks:
        raise RuntimeError("report.docxxml 为空")
    return chunks


def parse_anchors(text: str) -> dict[str, str]:
    """从源文件提取锚点 → 前一个正文块的文本片段(跳过其他锚点行)。"""
    lines = [l for l in text.split("\n") if l.strip()]
    anchors = {}
    for i, line in enumerate(lines):
        m = re.fullmatch(r"\[\[figure-anchor:([^\]]+)\]\]", line.strip())
        if m:
            j = i - 1
            while j >= 0 and re.fullmatch(r"\[\[figure-anchor:[^\]]+\]\]", lines[j].strip()):
                j -= 1
            if j < 0:
                raise RuntimeError(f"锚点 {m.group(1)} 之前没有正文块")
            prev = re.sub(r"<[^>]+>", " ", lines[j])
            prev = re.sub(r"\s+", " ", prev).strip()
            anchors[m.group(1)] = prev[-60:]  # 用段尾片段匹配,避开开头重复措辞
    return anchors


def find_block_id(content: str, text_frag: str) -> str:
    """在 with-ids 抓取的 XML 内容里找到包含文本片段的块的 id。

    按顶层块切开,块内去标签+归一空白后匹配(抓取结果会在 <b> 等内联标签
    周围插入额外空格,直接子串匹配会失败)。
    """
    frag = re.sub(r"\s+", " ", text_frag).strip()
    for m in re.finditer(r'<(\w+) id="([^"]+)"[^>]*>(.*?)</\1>', content, re.S):
        plain = re.sub(r"<[^>]+>", " ", m.group(3))
        plain = re.sub(r"\s+", " ", plain)
        if frag in plain:
            return m.group(2)
    raise RuntimeError(f"正文中找不到锚点前驱文本: …{frag[-40:]}")


def fetch_content(cli: str, doc: str, cwd: Path) -> str:
    d = run_cli(cli, ["docs", "+fetch", "--doc", doc, "--as", "bot", "--detail", "with-ids"], cwd)
    return d["data"]["document"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", help="运行 id,如 2026-08-03-vtla-vipe-spatialcli")
    ap.add_argument("--cli", default=DEFAULT_CLI, help="lark-cli 调用前缀")
    ap.add_argument("--user-id", default=None, help="IM 收件人 open_id;不给则不通知")
    ap.add_argument("--im-text", default=None, help="自定义私信文本(默认:报告标题+URL)")
    ap.add_argument("--dry-run", action="store_true", help="只分段并打印计划,不写入")
    args = ap.parse_args()

    cwd = Path.cwd()
    run_dir = cwd / "runs" / args.run_id
    report = run_dir / "report.docxxml"
    if not report.is_file():
        sys.exit(f"找不到 {report}")

    text = report.read_text()
    chunks = chunk_docxxml(text)
    anchors = parse_anchors(text)

    figures = {}
    manifest = run_dir / "assets" / "figures.json"
    if manifest.is_file():
        figures = json.loads(manifest.read_text())
        unknown = set(figures) - set(anchors)
        if unknown:
            sys.exit(f"figures.json 里的锚点在报告中不存在: {unknown}")

    print(f"分段: {len(chunks)} 段 ({[len(c.encode()) for c in chunks]})")
    print(f"锚点: {len(anchors)} 个,清单配图: {sum(len(v) if isinstance(v, list) else 1 for v in figures.values())} 张")
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="push-report-"))
        for i, c in enumerate(chunks, 1):
            (tmp / f"c{i}.xml").write_text(c)
        print(f"dry-run:分段已写入 {tmp},未推送")
        return

    # 1. 创建 + 分段追加
    with tempfile.TemporaryDirectory(prefix="push-report-", dir="drafts") as tmp_str:
        tmp = Path(tmp_str)
        paths = []
        for i, c in enumerate(chunks, 1):
            p = tmp / f"c{i}.xml"
            p.write_text(c)
            paths.append(p.relative_to(cwd))
        d = run_cli(args.cli, ["docs", "+create", "--as", "bot", "--content", f"@{paths[0]}"], cwd)
        doc = d["data"]["document"]["document_id"]
        url = d["data"]["document"]["url"]
        print(f"创建: {doc} {url}")
        for i, p in enumerate(paths[1:], 2):
            run_cli(args.cli, ["docs", "+update", "--doc", doc, "--as", "bot",
                               "--command", "append", "--content", f"@{p}"], cwd)
            print(f"追加 c{i}/{len(paths)} ok")

    # 2. 锚点插图
    inserted = 0
    if figures:
        for anchor, entries in figures.items():
            if isinstance(entries, dict):
                entries = [entries]
            target = find_block_id(fetch_content(args.cli, doc, cwd), anchors[anchor])
            for entry in entries:
                f = run_dir / entry["file"]
                rel = f.relative_to(cwd)
                d = run_cli(args.cli, ["docs", "+media-insert", "--doc", doc, "--as", "bot",
                                       "--file", str(rel), "--caption", entry["caption"]], cwd)
                img_id = d["data"]["block_id"]
                run_cli(args.cli, ["docs", "+update", "--doc", doc, "--as", "bot",
                                   "--command", "block_move_after",
                                   "--block-id", target, "--src-block-ids", img_id], cwd)
                target = img_id  # 链式:下一张移到本张之后
                inserted += 1
                print(f"插图 {anchor}: {entry['file']} -> {img_id}")
        # 3. 验证 img 数量
        n_imgs = len(re.findall(r"<img\b", fetch_content(args.cli, doc, cwd)))
        print(f"验证: 文档现有 {n_imgs} 张图(本次插入 {inserted})")
        if n_imgs < inserted:
            sys.exit("img 数量少于插入数,可能有孤儿图,请人工检查")

    # 4. IM 通知
    if args.user_id:
        msg = args.im_text or f"Paper Scout 巡航报告 {args.run_id.split('-')[0]}-… 已交付:\n{url}"
        run_cli(args.cli, ["im", "+messages-send", "--as", "bot",
                           "--user-id", args.user_id, "--text", msg], cwd)
        print("IM 已发送")

    print(f"DONE {url}")


if __name__ == "__main__":
    main()
