# stage-3 LLM Query & API Handling

Sending chat to the model, retries, error classification, usage/latency metrics.


## L3 `Terminus2._get_error_response_type` — `terminus_2.py:488-500`
- signature: `def _get_error_response_type(self) -> str:`
- assignment provenance: rule
- reads state: `_parser_name`
- callers: `Terminus2._run_agent_loop`

## L3 `Terminus2._get_completion_confirmation_message` — `terminus_2.py:502-523`
- signature: `def _get_completion_confirmation_message(self, terminal_output: str) -> str:`
- assignment provenance: rule
- reads state: `_parser_name`
- callers: `Terminus2._run_agent_loop`

## L3 `Terminus2._extract_usage_metrics` — `terminus_2.py:633-650`
- signature: `def _extract_usage_metrics(self, usage_info) -> tuple[int, int, int, float]:`
- assignment provenance: rule
- callers: `Terminus2._save_subagent_trajectory`

## L3 `Terminus2._track_api_request_time` — `terminus_2.py:652-660`
- signature: `def _track_api_request_time(self, start_time: float) -> None:`
- assignment provenance: rule
- reads state: `_api_request_times`
- callers: `Terminus2._run_subagent`

## L3 `Terminus2._query_llm` — `terminus_2.py:995-1155`
- signature: `async def _query_llm( self, chat: Chat, prompt: str, original_instruction: str = "", session: TmuxSession | None = None, ) -> LLMResponse:`
- decorators: `retry(stop=stop_after_attempt(3), retry=retry_if_not_exception_type(ContextLengthExceededError) & retry_if_exception_type(Exception), reraise=True)`
- assignment provenance: rule
- reads state: `_api_request_times`, `_enable_summarize`, `_llm`, `_llm_call_kwargs`, `_parser`, `logger`
- writes state: `_pending_handoff_prompt`, `_pending_subagent_refs`
- callers: `Terminus2._handle_llm_interaction`, `Terminus2._query_llm`
- calls: `Terminus2._query_llm`, `Terminus2._summarize`, `Terminus2._unwind_messages_to_free_tokens`
