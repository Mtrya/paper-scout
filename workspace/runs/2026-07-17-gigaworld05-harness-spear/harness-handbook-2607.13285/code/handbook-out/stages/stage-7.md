# stage-7 Context & Token Management

Token counting, output capping, proactive summarization, message unwinding.


## L3 `Terminus2._count_total_tokens` — `terminus_2.py:525-529`
- signature: `def _count_total_tokens(self, chat: Chat) -> int:`
- assignment provenance: rule
- reads state: `_model_name`
- callers: `Terminus2._check_proactive_summarization`, `Terminus2._unwind_messages_to_free_tokens`

## L3 `Terminus2._limit_output_length` — `terminus_2.py:531-567`
- signature: `def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:`
- assignment provenance: rule
- callers: `Terminus2._execute_commands`, `Terminus2._run_agent_loop`, `Terminus2.run`

## L3 `Terminus2._unwind_messages_to_free_tokens` — `terminus_2.py:569-593`
- signature: `def _unwind_messages_to_free_tokens( self, chat: Chat, target_free_tokens: int = 4000 ) -> None:`
- assignment provenance: rule
- reads state: `_llm`, `logger`
- callers: `Terminus2._query_llm`
- calls: `Terminus2._count_total_tokens`

## L3 `Terminus2._summarize` — `terminus_2.py:741-955`
- signature: `async def _summarize( self, chat: Chat, original_instruction: str, session: TmuxSession ) -> tuple[str, list[SubagentTrajectoryRef] | None]:`
- assignment provenance: rule
- reads state: `_model_name`, `_session_id`, `_summarization_count`
- writes state: `_summarization_count`
- callers: `Terminus2._check_proactive_summarization`, `Terminus2._query_llm`
- calls: `Terminus2._prepare_copied_trajectory_steps`, `Terminus2._run_subagent`, `TmuxSession.capture_pane`

## L3 `Terminus2._check_proactive_summarization` — `terminus_2.py:957-981`
- signature: `async def _check_proactive_summarization( self, chat: Chat, original_instruction: str, session: TmuxSession ) -> tuple[str, list[SubagentTrajectoryRef] | None] `
- assignment provenance: rule
- reads state: `_llm`, `_proactive_summarization_threshold`, `logger`
- callers: `Terminus2._run_agent_loop`
- calls: `Terminus2._count_total_tokens`, `Terminus2._summarize`

## L3 `Terminus2._split_trajectory_on_summarization` — `terminus_2.py:1862-1887`
- signature: `def _split_trajectory_on_summarization(self, handoff_prompt: str) -> None:`
- assignment provenance: rule-ambiguous(stage-7,stage-8)
- reads state: `_chat`, `_session_id`, `_summarization_count`
- writes state: `_session_id`, `_trajectory_steps`
- callers: `Terminus2._run_agent_loop`
- calls: `Terminus2._convert_chat_messages_to_steps`, `Terminus2._dump_trajectory_with_continuation_index`
