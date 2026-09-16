#!/bin/bash
# 实例重启后重建本地环境(幂等)。用法: bash rebuild_env.sh
set -u
WR=/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/ky26021/embodied-research/syncworld
if [ -x /root/sw-env/bin/python ]; then echo "env exists"; exit 0; fi
[ -d /root/uv-cache ] || cp -a $WR/cache/uv /root/uv-cache
cd $WR/SyncWorld
UV_CACHE_DIR=/root/uv-cache UV_PROJECT_ENVIRONMENT=/root/sw-env \
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
$WR/bin/uv sync --frozen --extra train --group=cu130-train
echo REBUILD_DONE
