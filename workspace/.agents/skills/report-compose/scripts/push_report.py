#!/usr/bin/env python3
"""Paper Scout 报告飞书推送:分段推送 + 锚点插图 + 验证 + IM 通知。

把一次巡航的 `runs/<run-id>/report.docxxml` 发布为飞书文档:
1. 按顶层块边界(每元素一行)切成 ≤5.8KB 的段,首段创建文档,其余
   `docs +update --command append` 逐段追加(长文档单次写入会被静默截断)。
2. 若存在 `runs/<run-id>/assets/figures.json`,按锚点插图:
   锚点 `[[figure-anchor:<name>]]` 独占段落在导入时**不**会被飞书丢弃(历史上
   曾假设会丢弃,导致锚点文本残留在文档里),因此插图采用块位置法——从源文件
   取锚点的前一个段落文本,在带块 id 的抓取结果里定位该段落,
   `docs +media-insert` 插图到末尾,再 `block_move_after` 移到位;
   同一锚点多张图按清单顺序链式移动。全部插完后,显式抓取锚点段落块 id
   并 `block_delete` 删除,再核验文档无 `figure-anchor` 残留。
3. 插图前对每张图跑白边机械检查(check_image_whitespace):单边空白占比
   >8% 时自动裁剪到临时副本再上传(不动运行包原件),整图 >60% 空白直接失败。
4. 重新抓取核对 img 数量与渲染宽度(width=100 且 scale>4 视为可疑,告警)。
5. 可选:给用户发 IM 私信(--user-id,用 --text 让 URL 展开成文档卡片)。

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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_image_whitespace as ciw  # noqa: E402

CHUNK_LIMIT = 5800  # 字节;单次 --content 超过 ~10KB 会被静默截断,留足余量
DEFAULT_CLI = "npx --yes @larksuite/cli"
WS_CROP = 0.08   # 单边空白占比超过此值即裁剪后上传
WS_FAIL = 0.60   # 整图空白占比超过此值视为坏图,拒绝推送


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
    """按非空行(每个顶层块一行)切 ≤CHUNK_LIMIT 字节的段,不跨块。

    table/pre/callout 等多行元素整体视为不可分单元:历史上一张表格恰好
    跨段边界时,后半段 tbody 会被飞书导入器压成表格外的裸段落。
    """
    lines = [l for l in text.split("\n") if l.strip()]
    units: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"<(\w+)[ >/]", line)
        tag = m.group(1) if m else ""
        if tag in ("table", "pre", "callout") and f"</{tag}>" not in line:
            buf, j = line, i + 1
            while j < len(lines) and f"</{tag}>" not in lines[j]:
                buf += "\n" + lines[j]
                j += 1
            if j < len(lines):
                buf += "\n" + lines[j]
                j += 1
            units.append(buf)
            i = j
        else:
            units.append(line)
            i += 1
    for u in units:
        if len(u.encode()) > CHUNK_LIMIT:
            raise RuntimeError(
                f"多行元素超过单段上限 {CHUNK_LIMIT} 字节,请拆成多个元素: {u[:60]}..."
            )
    chunks, cur = [], ""
    for u in units:
        if cur and len((cur + u + "\n").encode()) > CHUNK_LIMIT:
            chunks.append(cur)
            cur = ""
        cur += u + "\n"
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

    # 2. 锚点插图(插图前先做白边机械检查;坏图直接拒绝)
    inserted = 0
    if figures:
        ws_tmp = tempfile.TemporaryDirectory(prefix="push-report-ws-", dir="drafts")
        ws_dir = Path(ws_tmp.name)
        for anchor, entries in figures.items():
            if isinstance(entries, dict):
                entries = [entries]
            target = find_block_id(fetch_content(args.cli, doc, cwd), anchors[anchor])
            for entry in entries:
                f = run_dir / entry["file"]
                fracs, _ = ciw.measure(f)
                if max(fracs) > WS_FAIL:
                    sys.exit(f"{entry['file']} 整图空白占比 {max(fracs):.0%} > {WS_FAIL:.0%},疑似坏图,拒绝推送")
                if max(fracs) > WS_CROP:
                    cropped = ws_dir / f.name
                    shutil.copy(f, cropped)
                    ciw.crop(cropped, pad=8)
                    print(f"白边裁剪 {entry['file']}: top {fracs[0]:.1%} bottom {fracs[1]:.1%} "
                          f"left {fracs[2]:.1%} right {fracs[3]:.1%} -> 使用裁剪副本上传")
                    f = cropped
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
        ws_tmp.cleanup()
        # 3. 显式删除锚点段落(飞书导入不会丢弃它们,历史上曾假设会丢弃)
        content = fetch_content(args.cli, doc, cwd)
        anchor_ids = re.findall(r'<p id="([^"]+)">\[\[figure-anchor:[^\]]+\]\]</p>', content)
        if anchor_ids:
            run_cli(args.cli, ["docs", "+update", "--doc", doc, "--as", "bot",
                               "--command", "block_delete",
                               "--block-id", ",".join(anchor_ids)], cwd)
            print(f"锚点段落已删除: {len(anchor_ids)} 个")
        # 4. 验证 img 数量、锚点残留与渲染宽度
        content = fetch_content(args.cli, doc, cwd)
        n_imgs = len(re.findall(r"<img\b", content))
        print(f"验证: 文档现有 {n_imgs} 张图(本次插入 {inserted})")
        if n_imgs < inserted:
            sys.exit("img 数量少于插入数,可能有孤儿图,请人工检查")
        if "figure-anchor" in content:
            sys.exit("锚点段落删除后仍有 figure-anchor 残留,请人工检查")
        for m in re.finditer(r'<img\b[^>]*\bscale="([\d.]+)"[^>]*\bwidth="(\d+)"', content):
            scale, width = float(m.group(1)), int(m.group(2))
            if width <= 100 and scale > 4:
                print(f"警告: 图片 {m.group(0)[:80]}… 渲染宽度可疑(width={width} scale={scale}),请人工核对显示效果")

    # 4. IM 通知
    if args.user_id:
        msg = args.im_text or f"Paper Scout 巡航报告 {args.run_id.split('-')[0]}-… 已交付:\n{url}"
        run_cli(args.cli, ["im", "+messages-send", "--as", "bot",
                           "--user-id", args.user_id, "--text", msg], cwd)
        print("IM 已发送")

    print(f"DONE {url}")


if __name__ == "__main__":
    main()
