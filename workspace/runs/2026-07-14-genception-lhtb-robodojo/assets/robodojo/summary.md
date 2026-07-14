# RoboDojo leaderboard probe summary

- Simulation policies parsed: 30
- Real-world policies parsed: 10
- Best simulation policy (avg success): Hy-Embodied-0.5-VLA(Zhang et al., [2026a](https://arxiv.org/html/2607.04434#bib.bib72) @ 8.80%
- Human teleop simulation avg success: 76.03%
- Overall sim gap to human: 67.23 pp
- Best real-world policy (overall success): \pi_{0.5}(Intelligence et al., [2025](https://arxiv.org/html/2607.04434#bib.bib22) @ 12.80%
- Overall real gap to human: 87.20 pp

## Largest per-dimension sim gaps to human teleop

- **Generalization**: human 87.83%, best (Spatial Forcing(Li et al., [2025a](https://arxiv.org/html/2607.04434#bib.bib31)) 9.33% → gap 78.50 pp (89.4% relative)
- **Precision**: human 64.00%, best (X-VLA(Zheng et al., [2025](https://arxiv.org/html/2607.04434#bib.bib79)) 12.00% → gap 52.00 pp (81.2% relative)
- **Long-Horizon**: human 74.25%, best (Hy-Embodied-0.5-VLA(Zhang et al., [2026a](https://arxiv.org/html/2607.04434#bib.bib72)) 14.92% → gap 59.33 pp (79.9% relative)
- **Memory**: human 74.33%, best (Hy-Embodied-0.5-VLA(Zhang et al., [2026a](https://arxiv.org/html/2607.04434#bib.bib72)) 12.11% → gap 62.22 pp (83.7% relative)
- **Open**: human 79.75%, best (\pi_{0.5}(Intelligence et al., [2025](https://arxiv.org/html/2607.04434#bib.bib22)) 1.67% → gap 78.08 pp (97.9% relative)
- **Average**: human 76.03%, best (Hy-Embodied-0.5-VLA(Zhang et al., [2026a](https://arxiv.org/html/2607.04434#bib.bib72)) 8.80% → gap 67.23 pp (88.4% relative)

## Sim-vs-real overlap (policies in both leaderboards)

- \pi_{0.5}(Intelligence et al., [2025](https://arxiv.org/html/2607.04434#bib.bib22): sim 6.91% (rank 3) → real 12.80% (rank 1); drop -5.89 pp
- X-VLA(Zheng et al., [2025](https://arxiv.org/html/2607.04434#bib.bib79): sim 6.52% (rank 4) → real 3.30% (rank 5); drop 3.22 pp
- Xiaomi-Robotics-0(Cai et al., [2026c](https://arxiv.org/html/2607.04434#bib.bib9): sim 4.18% (rank 5) → real 3.90% (rank 4); drop 0.28 pp
- StarVLA-\alpha(Ye et al., [2026b](https://arxiv.org/html/2607.04434#bib.bib64): sim 3.24% (rank 8) → real 1.70% (rank 8); drop 1.54 pp
- GalaxeaVLA (G0): sim 2.96% (rank 9) → real 4.40% (rank 3); drop -1.44 pp
- \pi_{0}(Black et al., [2024](https://arxiv.org/html/2607.04434#bib.bib5): sim 1.53% (rank 15) → real 1.70% (rank 7); drop -0.17 pp
- GR00T-N1.7(Bjorck et al., [2025](https://arxiv.org/html/2607.04434#bib.bib4): sim 1.31% (rank 16) → real 1.70% (rank 6); drop -0.39 pp
- InternVLA-A1(Cai et al., [2026b](https://arxiv.org/html/2607.04434#bib.bib8): sim 1.08% (rank 17) → real 7.20% (rank 2); drop -6.12 pp
- Spirit v1.5([Team et al.,](https://arxiv.org/html/2607.04434#bib.bib53): sim 0.14% (rank 23) → real 0.60% (rank 9); drop -0.46 pp
- DM0(Yu et al., [2026a](https://arxiv.org/html/2607.04434#bib.bib66): sim 0.05% (rank 27) → real 0.00% (rank 10); drop 0.05 pp

## Artifacts produced

- `fig1.png`
- `fig10.png`
- `fig11.png`
- `fig2.png`
- `fig3.png`
- `fig4.png`
- `fig5.png`
- `fig6.png`
- `fig7.png`
- `fig8.png`
- `fig9.png`
- `real_ranking.csv`
- `sim_dimension_gaps.png`
- `sim_gaps.csv`
- `sim_ranking.csv`
- `sim_real_overlap.csv`
- `sim_real_scatter.png`
- `sim_top10_success.png`