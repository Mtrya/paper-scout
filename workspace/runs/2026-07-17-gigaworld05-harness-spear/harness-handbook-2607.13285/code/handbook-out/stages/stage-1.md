# stage-1 Initialization & Config

Agent construction, LLM client setup, model-info resolution, API surface (name/version).


## L3 `Terminus2._init_llm` — `terminus_2.py:82-154`
- signature: `def _init_llm( self, llm_backend: LLMBackend | str, model_name: str, temperature: float | None, collect_rollout_details: bool, llm_kwargs: dict[str, Any] | None`
- assignment provenance: rule
- callers: `Terminus2.__init__`

## L3 `Terminus2.__init__` — `terminus_2.py:156-344`
- signature: `def __init__( self, logs_dir: Path, model_name: str | None = None, max_turns: int | None = None, parser_name: str = "json", api_base: str | None = None, tempera`
- assignment provenance: rule
- reads state: `_trajectory_config`, `logger`
- writes state: `_api_request_times`, `_chat`, `_collect_rollout_details`, `_context`, `_enable_summarize`, `_interleaved_thinking`, `_last_response_model_name`, `_linear_history`, `_llm`, `_llm_call_kwargs`, `_llm_kwargs`, `_max_episodes`, `_model_name`, `_n_episodes`, `_parser`, `_parser_name`, `_pending_completion`, `_pending_handoff_prompt`, `_pending_subagent_refs`, `_proactive_summarization_threshold`, `_prompt_template`, `_reasoning_effort`, `_record_terminal_session`, `_save_raw_content_in_trajectory`, `_session`, `_session_id`, `_store_all_messages`, `_subagent_metrics`, `_subagent_rollout_details`, `_summarization_count`, `_temperature`, `_timeout_template`, `_tmux_pane_height`, `_tmux_pane_width`, `_trajectory_config`, `_trajectory_steps`, `_user_provided_session_id`
- calls: `Terminus2._get_parser`, `Terminus2._get_prompt_template_path`, `Terminus2._get_timeout_template_path`, `Terminus2._init_llm`, `Terminus2._resolve_model_info`

## L3 `Terminus2._resolve_model_info` — `terminus_2.py:346-358`
- signature: `def _resolve_model_info( self, model_name: str | None, provided_model_info: dict[str, Any] | None ) -> dict[str, Any] | None:`
- assignment provenance: rule
- reads state: `logger`
- callers: `Terminus2.__init__`

## L3 `Terminus2.name` — `terminus_2.py:362-363`
- signature: `def name() -> str:`
- decorators: `staticmethod`, `override`
- assignment provenance: rule
- callers: `Terminus2._dump_trajectory_with_continuation_index`, `Terminus2.setup`

## L3 `Terminus2.version` — `terminus_2.py:366-367`
- signature: `def version(self) -> str | None:`
- decorators: `override`
- assignment provenance: rule
- callers: `Terminus2._dump_trajectory_with_continuation_index`, `Terminus2._save_subagent_trajectory`
