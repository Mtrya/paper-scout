# Memory-arbitration results

## Probe 1: detection P/R/F1 (pooled entry-level; per-seed F1 mean+-std)

| model | mode | regime | P | R | F1 | per-seed F1 | parse-fail |
|---|---|---|---|---|---|---|---|
| glm41v9b | text | L1 | 1.000 | 1.000 | 1.000 | n/a | 0/7 |
| glm41v9b | text | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/1 |
| glm41v9b | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| glm41v9b | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| qwen3vl8b | text | L1 | 1.000 | 1.000 | 1.000 | 1.000+-0.000 | 0/28 |
| qwen3vl8b | text | L2 | 1.000 | 0.973 | 0.986 | 0.987+-0.026 | 0/28 |
| qwen3vl8b | vision | L1 | 0.108 | 0.295 | 0.158 | 0.162+-0.083 | 0/126 |
| qwen3vl8b | vision | L2 | 0.214 | 0.423 | 0.284 | 0.279+-0.121 | 0/124 |
| smoke_g | text | L1 | 1.000 | 1.000 | 1.000 | n/a | 5/7 |
| smoke_g | text | L2 | 0.000 | 0.000 | 0.000 | n/a | 1/1 |
| smoke_g | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_g | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_g3 | text | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_g3 | text | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_g3 | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_g3 | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_q | text | L1 | 1.000 | 1.000 | 1.000 | n/a | 0/7 |
| smoke_q | text | L2 | 1.000 | 1.000 | 1.000 | n/a | 0/7 |
| smoke_q | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_q | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qv | text | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qv | text | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qv | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/4 |
| smoke_qv | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qvv | text | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qvv | text | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| smoke_qvv | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/4 |
| smoke_qvv | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |

## Probe 2: image ablation (vision mode; per-entry judgment flips vs correct image, reasoning similarity)

| model | regime | comparison | flip rate | reason sim (difflib) | reason sim (jaccard) | flag rate correct | flag rate ablated | n |
|---|---|---|---|---|---|---|---|---|
| qwen3vl8b | L1 | correct vs blank | 0.248 | 0.742 | 0.539 | 0.248 | 0.050 | 960 |
| qwen3vl8b | L1 | correct vs mismatch | 0.287 | 0.755 | 0.548 | 0.248 | 0.267 | 960 |
| qwen3vl8b | L2 | correct vs blank | 0.279 | 0.726 | 0.514 | 0.271 | 0.067 | 960 |
| qwen3vl8b | L2 | correct vs mismatch | 0.306 | 0.761 | 0.561 | 0.273 | 0.241 | 951 |
| smoke_qv | L1 | correct vs blank | 0.100 | 0.763 | 0.555 | 0.100 | 0.000 | 40 |
| smoke_qv | L1 | correct vs mismatch | 0.075 | 0.796 | 0.636 | 0.100 | 0.025 | 40 |
| smoke_qvv | L1 | correct vs blank | 0.000 | 0.721 | 0.516 | 0.025 | 0.025 | 40 |
| smoke_qvv | L1 | correct vs mismatch | 0.100 | 0.779 | 0.562 | 0.025 | 0.075 | 40 |

## Probe 3: arbitration weights (text mode; P(judge stale) by factor level)

| model | factor | level | P(stale\|conflict) | P(stale\|control) | n_conflict | n_control |
|---|---|---|---|---|---|---|
| glm41v9b | F1 | definitely | 1.000 | 0.050 | 20 | 20 |
| glm41v9b | F1 | probably | 1.000 | 0.000 | 20 | 20 |
| glm41v9b | F1 | neutral | 1.000 | 0.000 | 20 | 20 |
| glm41v9b | F2 | now | 1.000 | 0.000 | 20 | 20 |
| glm41v9b | F2 | ago | 1.000 | 0.000 | 20 | 20 |
| glm41v9b | F2 | none | 1.000 | 0.000 | 20 | 19 |
| glm41v9b | F3 | frequent | 1.000 | 0.050 | 20 | 20 |
| glm41v9b | F3 | rare | 1.000 | 0.150 | 20 | 20 |
| glm41v9b | F3 | none | 0.950 | 0.000 | 20 | 20 |
| glm41v9b | F4 | accurate | 1.000 | 0.000 | 20 | 20 |
| glm41v9b | F4 | plain | 1.000 | 0.053 | 20 | 19 |

| qwen3vl8b | F1 | definitely | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F1 | probably | 0.950 | 0.000 | 20 | 20 |
| qwen3vl8b | F1 | neutral | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F2 | now | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F2 | ago | 0.950 | 0.000 | 20 | 20 |
| qwen3vl8b | F2 | none | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F3 | frequent | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F3 | rare | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F3 | none | 0.950 | 0.000 | 20 | 20 |
| qwen3vl8b | F4 | accurate | 1.000 | 0.000 | 20 | 20 |
| qwen3vl8b | F4 | plain | 0.950 | 0.000 | 20 | 20 |


| smoke_g3 | F1 | definitely | 1.000 | 0.000 | 4 | 4 |



