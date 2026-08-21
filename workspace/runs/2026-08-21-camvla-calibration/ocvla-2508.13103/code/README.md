# 本线程无保留代码产物

本线程是 OC-VLA(2508.13103)的精读 + 代码核验线程,核验结论(核心证据)都在本目录的 `README.md`:

- 官方仓库 `ZTY0213/OC-VLA` 是空壳:LICENSE + README 共 2 个文件,正文即「Code 见 Dita 仓库」——这一点本身就是核验发现,无需保存该仓库。
- OC-VLA 的实际实现位于 Dita 仓库的 `use_baseframe_action=False` 分支;Dita 稀疏克隆(74M)在工作区 `code/dita/`(gitignored),未保留副本。
- 变换实现核验点:`utils/data_utils.py::process_traj_v3`、`Dataset_Sim/SimDataset.py`、`scripts/close_loop_eval_diffusion.py` 的 `cal_action_from_pose` 调用路径,均已逐行核验并记录在 README。

如需复跑核验,克隆 `github.com/ZTY0213/OC-VLA` 与 `github.com/RoboDita/Dita` 即可,证据链见线程 README 第 4 节。
