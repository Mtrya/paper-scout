# stage-9 Subagent Handoff

Spawning subagents with summarized context, aggregating their metrics.


## L3 `Terminus2._collect_subagent_rollout_detail` — `terminus_2.py:595-617`
- signature: `def _collect_subagent_rollout_detail(self, response: LLMResponse) -> None:`
- assignment provenance: rule
- reads state: `_collect_rollout_details`, `_subagent_rollout_details`
- callers: `Terminus2._run_subagent`

## L3 `Terminus2._update_subagent_metrics` — `terminus_2.py:619-631`
- signature: `def _update_subagent_metrics(self, usage_info) -> None:`
- assignment provenance: rule
- reads state: `_subagent_metrics`
- callers: `Terminus2._run_subagent`
