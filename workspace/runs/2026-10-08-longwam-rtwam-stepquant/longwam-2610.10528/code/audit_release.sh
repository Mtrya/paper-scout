#!/usr/bin/env bash
# Long-WAM 发布资产核验脚本(arXiv 2610.10528)
# 对实验台克隆 code/LongLive/Long-WAM/ 跑核验查询,输出写入 audit_output.txt。
# 用法: bash audit_release.sh [path-to-Long-WAM]   (默认 ../../../../../code/LongLive/Long-WAM)
set -u
LW="${1:-../../../../code/LongLive/Long-WAM}"
{
  echo "# Long-WAM 发布资产核验输出 ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "# 目标: $LW (HEAD $(git -C "$LW" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  echo

  echo "## 1. 已发布 checkpoint(configs/checkpoints.yaml 的 repo_id)"
  grep -c "repo_id: " "$LW/configs/checkpoints.yaml" 2>/dev/null
  grep -n "repo_id: null" "$LW/configs/checkpoints.yaml" 2>/dev/null
  echo

  echo "## 2. 未发布初始化资产的 TBD 标注"
  grep -rn "TBD" "$LW/docs/TRAINING.md" 2>/dev/null | head -5
  grep -rn "TBD" "$LW/docs/MERGE.md" 2>/dev/null | head -5
  echo

  echo "## 3. 论文成功率数字在仓库中的命中(带数字边界防误命中)"
  grep -rnE "[^0-9.](63\.3|78\.7|54\.4)[^0-9]" "$LW" --include="*.md" --include="*.py" --include="*.yaml" 2>/dev/null | grep -v ".git" | head -5
  echo "(无输出 = 论文数字不入库)"
  echo

  echo "## 4. 107.4ms 延迟的出处与限定"
  grep -n -B2 -A4 "107.4" "$LW/infra/README.md" 2>/dev/null | head -20
  echo

  echo "## 5. 双向对照配方的排除声明"
  grep -n "imagination-ablation\|ablation" "$LW/docs/MERGE.md" 2>/dev/null | head -5
  echo

  echo "## 6. 流式推理强制因果 mask 的位置"
  grep -n "per_frame_causal" "$LW/src/longwam/models/wan22/longwam_base.py" 2>/dev/null | head -5
  echo

  echo "## 7. FastWAM 继承文件计数(宽口径 = 引用 FastWAM;窄口径 = 带 provenance 标注)"
  echo -n "宽口径: "; grep -rln "FastWAM\|fastwam" "$LW/src" 2>/dev/null | wc -l
  echo -n "窄口径: "; grep -rln "Copyright.*FastWAM\|from.*FastWAM\|FastWAM (" "$LW/src" 2>/dev/null | wc -l
} | tee "$(dirname "$0")/audit_output.txt"
