# stage-10 Per-Run Lifecycle & Reset

run()/perform_task entry points and per-run state reset.


## L3 `Terminus2._build_skills_section` — `terminus_2.py:435-486`
- signature: `async def _build_skills_section(self, environment: BaseEnvironment) -> str | None:`
- assignment provenance: callgraph-propagation
- reads state: `skills_dir`
- callers: `Terminus2.run`
- calls: `Terminus2._parse_skill_frontmatter`

## L3 `Terminus2._reset_per_run_state` — `terminus_2.py:1542-1556`
- signature: `def _reset_per_run_state(self) -> None:`
- assignment provenance: rule
- reads state: `_user_provided_session_id`
- writes state: `_api_request_times`, `_n_episodes`, `_pending_completion`, `_pending_handoff_prompt`, `_pending_subagent_refs`, `_session_id`, `_subagent_metrics`, `_subagent_rollout_details`, `_summarization_count`, `_trajectory_steps`
- callers: `Terminus2.run`

## L3 `Terminus2.run` — `terminus_2.py:1559-1643`
- signature: `async def run( self, instruction: str, environment: BaseEnvironment, context: AgentContext, ) -> None:`
- decorators: `override`
- assignment provenance: rule
- reads state: `_api_request_times`, `_chat`, `_interleaved_thinking`, `_llm`, `_n_episodes`, `_prompt_template`, `_session`, `_store_all_messages`, `_subagent_metrics`, `_subagent_rollout_details`, `_summarization_count`, `_trajectory_steps`, `mcp_servers`
- writes state: `_chat`, `_context`
- calls: `Terminus2._build_skills_section`, `Terminus2._dump_trajectory`, `Terminus2._limit_output_length`, `Terminus2._reset_per_run_state`, `Terminus2._run_agent_loop`, `TmuxSession.get_incremental_output`
