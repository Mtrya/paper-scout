# stage-5 Main Agent Loop

The observe-decide-act iteration: episode bookkeeping, observation handling, command dispatch, and the completion gate that decides loop termination.


## L3 `_terminal_observation_source_call_id` — `terminus_2.py:61-66`
- signature: `def _terminal_observation_source_call_id( commands: list[Command], episode: int ) -> str | None:`
- assignment provenance: callgraph-propagation
- callers: `Terminus2._run_agent_loop`

## L3 `Terminus2._handle_llm_interaction` — `terminus_2.py:1157-1197`
- signature: `async def _handle_llm_interaction( self, chat: Chat, prompt: str, original_instruction: str = "", session: TmuxSession | None = None, ) -> tuple[list[Command], `
- assignment provenance: rule
- reads state: `_parser`, `logger`
- callers: `Terminus2._run_agent_loop`
- calls: `Terminus2._query_llm`

## L3 `Terminus2._execute_commands` — `terminus_2.py:1199-1229`
- signature: `async def _execute_commands( self, commands: list[Command], session: TmuxSession, ) -> tuple[bool, str]:`
- assignment provenance: rule
- reads state: `_timeout_template`
- callers: `Terminus2._run_agent_loop`
- calls: `Terminus2._limit_output_length`, `TmuxSession.get_incremental_output`, `TmuxSession.send_keys`

## L3 `Terminus2._run_agent_loop` — `terminus_2.py:1231-1540`
- signature: `async def _run_agent_loop( self, initial_prompt: str, chat: Chat, original_instruction: str = "", ) -> None:`
- assignment provenance: rule
- reads state: `_context`, `_enable_summarize`, `_linear_history`, `_max_episodes`, `_model_name`, `_pending_completion`, `_pending_handoff_prompt`, `_pending_subagent_refs`, `_save_raw_content_in_trajectory`, `_session`, `_trajectory_steps`, `logger`
- writes state: `_last_response_model_name`, `_n_episodes`, `_pending_completion`, `_pending_handoff_prompt`, `_pending_subagent_refs`
- callers: `Terminus2.run`
- calls: `Terminus2._check_proactive_summarization`, `Terminus2._dump_trajectory`, `Terminus2._execute_commands`, `Terminus2._get_completion_confirmation_message`, `Terminus2._get_error_response_type`, `Terminus2._handle_llm_interaction`, `Terminus2._limit_output_length`, `Terminus2._split_trajectory_on_summarization`, `TmuxSession.is_session_alive`, `_terminal_observation_source_call_id`
