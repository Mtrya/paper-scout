# 本线程无保留代码产物

本线程是动作接口横切(BARX / ContactFlow / AxisGuide)的精读 + 代码核验线程,核心证据都在本目录的 `README.md`:

- 代码核验对象是外部仓库,克隆在工作区 gitignored 的 `code/` 下:`code/barx/`(333M,完整克隆)与 `code/axisguide-check/`(19M,lerobot fork),未保留副本。核验结论(BARX 的 `AUX_TASK_QA_FUNCTIONS`/特权 trace、AxisGuide 的 `visual_cue_mode=basis_concat` 等)已逐点记录在 README 第 4 节,复跑只需重新克隆对应仓库。
- ContactFlow 无整合代码放出(投稿前状态),已如实记录。

如需复跑核验:克隆 `ajaysridhar0/barx`(MIT)与 `JiyunJang-24/AxisGuide-code`(Apache-2.0),核对点见线程 README。
