"""监督信号光谱对照 —— 三条线的监督来源 × 学习型组件成败。

产出本期跨主题观察的核心对照表(数字均带出处):
- RynnValue(08-12 巡航探针,官方 4B 权重八条件扰动):runs/2026-08-12-rynnvalue-shortcuts-ouroboros/rynnvalue-2608.09853/
- U-OPSD(本期探针):本线程代码 uopsd_probe.py / 远端结果
- NWAT(论文表 1-3):papers/llm-agents/marginal-token-value-2608.08389.md

用法:python signal_spectrum.py [--uopsd-base 0.xx --uopsd-final 0.xx]
"""
import argparse
import json

RYNN = {
    "system": "RynnValue 价值函数",
    "signal": "演示数据真实时间戳 → cost-to-go 回归",
    "signal_kind": "真值派生,干净、无歧义",
    "evidence": "shuffle ρ(v,剩余时间)=0.76 vs ρ(v,位置)=-0.03;回退检测 5.3s 上跳;多尺度校准良好",
    "verdict": "学习型价值函数成立、内容接地、防捷径",
}

GENI = {
    "system": "GeniWorld WAM 接口",
    "signal": "URDF 渲染的动作像素(构造的接口信号)",
    "signal_kind": "构造出来的干净接口",
    "evidence": "RoboTwin Clean-to-Random FID 13.08 vs Ctrl-World 21.66;5 步采样 FVD 仅退化 ~2%",
    "verdict": "接口越自明,模型越不需要学映射",
}

NWAT = {
    "system": "NWAT 剪枝价值模型",
    "signal": "LLM-judge rubric 分 + 引用召回",
    "signal_kind": "噪声代理,判官敏感",
    "evidence": "学习型控制器仅'探索性概念验证',打不过 MMR 等启发式;所有剪枝方法 KPR+KPC 未超基线",
    "verdict": "模糊效用信号下学习型组件平庸",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uopsd-base", type=float, default=None, help="base maj@8 (train prompts)")
    ap.add_argument("--uopsd-final", type=float, default=None, help="final maj@8 (train prompts)")
    ap.add_argument("--uopsd-split-wrong-amplify", type=float, default=None,
                    help="split_wrong wrong_agree delta (final - base)")
    args = ap.parse_args()

    uopsd = {
        "system": "U-OPSD 自蒸馏",
        "signal": "模型自身多数投票(伪金标)",
        "signal_kind": "清晰可构造,但不携带正确性",
        "evidence": (
            f"论文:13.3% 伪标金错;探针:base maj@8={args.uopsd_base} → final={args.uopsd_final}"
            if args.uopsd_base else
            "论文:13.3% 伪标金错;探针:见 eval_base.json / merged_*_metrics.json"),
        "verdict": "多数正确处巩固,系统性错误处失明/放大"
        + (f"(split_wrong wrong_agree Δ={args.uopsd_split_wrong_amplify:+.3f})"
           if args.uopsd_split_wrong_amplify is not None else ""),
    }
    rows = [RYNN, GENI, uopsd, NWAT]
    json.dump(rows, open("signal_spectrum.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
