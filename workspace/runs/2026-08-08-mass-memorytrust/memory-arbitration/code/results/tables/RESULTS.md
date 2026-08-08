# Memory-arbitration results

## Probe 1: detection P/R/F1 (pooled entry-level; per-seed F1 mean+-std)

| model | mode | regime | P | R | F1 | per-seed F1 | parse-fail |
|---|---|---|---|---|---|---|---|
| glm41v9b | text | L1 | 0.900 | 1.000 | 0.947 | 0.933+-0.103 | 3/37 |
| glm41v9b | text | L2 | 0.949 | 1.000 | 0.974 | 0.971+-0.039 | 1/35 |
| glm41v9b | vision | L1 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| glm41v9b | vision | L2 | 0.000 | 0.000 | 0.000 | n/a | 0/0 |
| qwen3vl8b | text | L1 | 1.000 | 1.000 | 1.000 | 1.000+-0.000 | 0/42 |
| qwen3vl8b | text | L2 | 0.964 | 0.982 | 0.973 | 0.975+-0.042 | 0/42 |
| qwen3vl8b | vision | L1 | 0.108 | 0.295 | 0.158 | 0.162+-0.083 | 0/126 |
| qwen3vl8b | vision | L2 | 0.214 | 0.423 | 0.284 | 0.279+-0.121 | 0/124 |

## Probe 2: image ablation (vision mode; per-entry judgment flips vs correct image, reasoning similarity)

| model | regime | comparison | flip rate | reason sim (difflib) | reason sim (jaccard) | flag rate correct | flag rate ablated | n |
|---|---|---|---|---|---|---|---|---|
| qwen3vl8b | L1 | correct vs blank | 0.248 | 0.742 | 0.539 | 0.248 | 0.050 | 960 |
| qwen3vl8b | L1 | correct vs mismatch | 0.287 | 0.755 | 0.548 | 0.248 | 0.267 | 960 |
| qwen3vl8b | L2 | correct vs blank | 0.279 | 0.726 | 0.514 | 0.271 | 0.067 | 960 |
| qwen3vl8b | L2 | correct vs mismatch | 0.306 | 0.761 | 0.561 | 0.273 | 0.241 | 951 |

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
