"""Ouroboros git archaeology figure: cumulative commits by author + BIBLE amendments."""
import subprocess
from collections import defaultdict
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]

def gitlog(fmt, path=None):
    cmd = ["git", "log", "--format=" + fmt, "--date=short", "--reverse"]
    if path:
        cmd += ["--", path]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd="code/ouroboros").stdout
    return [l for l in out.strip().splitlines() if l]

# cumulative commits by author class
daily = defaultdict(lambda: [0, 0])  # date -> [agent, human]
for line in gitlog("%ad|%an"):
    d, author = line.split("|", 1)
    y, m, dd = map(int, d.split("-"))
    daily[date(y, m, dd)][author == "Ouroboros"] += 1

days = sorted(daily)
agent_cum, human_cum, a, h = [], [], 0, 0
for d in days:
    a += daily[d][1]
    h += daily[d][0]
    agent_cum.append(a)
    human_cum.append(h)

# BIBLE.md amendments (agent-authored, after initial commit)
bible_days = []
for line in gitlog("%ad|%an|%s", "BIBLE.md")[1:]:
    d, author, _ = line.split("|", 2)
    if author == "Ouroboros":
        y, m, dd = map(int, d.split("-"))
        bible_days.append(date(y, m, dd))

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(days, [a + h for a, h in zip(agent_cum, human_cum)], color="#888", lw=1.2,
        label="总提交")
ax.plot(days, agent_cum, color="#c0392b", lw=2.2, label="agent 署名(Ouroboros)")
ax.fill_between(days, agent_cum, color="#c0392b", alpha=0.12)
for i, bd in enumerate(bible_days):
    ax.axvline(bd, color="#2980b9", lw=0.9, alpha=0.65, ls="--",
               label="agent 自修订 BIBLE.md" if i == 0 else None)
ax.set_ylabel("累计 commit 数")
ax.set_title("Ouroboros 仓库 947 个 commit:74% 由 agent 自己署名,BIBLE.md 被 agent 经门禁修订 12 次",
             fontsize=10.5)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(alpha=0.25)
ax.text(0.52, 0.42, f"agent {agent_cum[-1]} / 总数 {agent_cum[-1]+human_cum[-1]}",
        transform=ax.transAxes, fontsize=10, color="#c0392b")
fig.tight_layout()
fig.savefig("drafts/ouroboros-archaeology.png", dpi=170)
print("agent commits:", agent_cum[-1], "total:", agent_cum[-1] + human_cum[-1],
      "bible agent amendments:", len(bible_days), "span:", days[0], "->", days[-1])
