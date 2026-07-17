# stage-8 Trajectory & Recording

Step/trajectory persistence, asciinema recording, ATIF conversion.


## L3 `AsciinemaHandler.__init__` — `asciinema_handler.py:11-20`
- signature: `def __init__(self, markers: list[tuple[float, str]], recording_path: Path):`
- assignment provenance: rule
- writes state: `_markers`, `_recording_path`
- callers: `TmuxSession.stop`

## L3 `AsciinemaHandler.merge_markers` — `asciinema_handler.py:22-39`
- signature: `def merge_markers(self) -> None:`
- assignment provenance: rule
- reads state: `_markers`, `_recording_path`
- calls: `AsciinemaHandler._write_merged_recording`

## L3 `AsciinemaHandler._write_merged_recording` — `asciinema_handler.py:41-60`
- signature: `def _write_merged_recording(self, output_path: Path) -> None:`
- assignment provenance: rule
- reads state: `_markers`, `_recording_path`
- callers: `AsciinemaHandler.merge_markers`
- calls: `AsciinemaHandler._process_recording_line`, `AsciinemaHandler._write_remaining_markers`

## L3 `AsciinemaHandler._process_recording_line` — `asciinema_handler.py:62-90`
- signature: `def _process_recording_line( self, line: str, output_file: TextIO, marker_index: int, ) -> int:`
- assignment provenance: rule
- reads state: `_markers`
- callers: `AsciinemaHandler._write_merged_recording`
- calls: `AsciinemaHandler._write_marker`

## L3 `AsciinemaHandler._write_marker` — `asciinema_handler.py:92-96`
- signature: `def _write_marker(self, output_file: TextIO, marker: tuple[float, str]) -> None:`
- assignment provenance: rule
- callers: `AsciinemaHandler._process_recording_line`, `AsciinemaHandler._write_remaining_markers`

## L3 `AsciinemaHandler._write_remaining_markers` — `asciinema_handler.py:98-103`
- signature: `def _write_remaining_markers( self, output_file: TextIO, markers: list[tuple[float, str]] ) -> None:`
- assignment provenance: rule
- callers: `AsciinemaHandler._write_merged_recording`
- calls: `AsciinemaHandler._write_marker`

## L3 `Terminus2._run_subagent` — `terminus_2.py:662-739`
- signature: `async def _run_subagent( self, prompt: str, message_history: list[dict[str, Any]], steps: list[Step], session_id: str, agent_name: str, filename_suffix: str, su`
- assignment provenance: rule-ambiguous(stage-8,stage-9)
- reads state: `_llm`, `_llm_call_kwargs`
- callers: `Terminus2._summarize`
- calls: `Terminus2._append_subagent_response_step`, `Terminus2._collect_subagent_rollout_detail`, `Terminus2._save_subagent_trajectory`, `Terminus2._track_api_request_time`, `Terminus2._update_subagent_metrics`

## L3 `Terminus2._remove_metrics_from_copied_steps` — `terminus_2.py:1648-1665`
- signature: `def _remove_metrics_from_copied_steps(steps: list[Step]) -> None:`
- decorators: `staticmethod`
- assignment provenance: rule
- callers: `Terminus2._prepare_copied_trajectory_steps`

## L3 `Terminus2._prepare_copied_trajectory_steps` — `terminus_2.py:1667-1681`
- signature: `def _prepare_copied_trajectory_steps( self, steps_to_include: int ) -> tuple[list[Step], int]:`
- assignment provenance: rule
- reads state: `_trajectory_steps`
- callers: `Terminus2._summarize`
- calls: `Terminus2._remove_metrics_from_copied_steps`

## L3 `Terminus2._append_subagent_response_step` — `terminus_2.py:1683-1733`
- signature: `def _append_subagent_response_step( self, steps: list[Step], step_id: int, response: LLMResponse, usage_info, subagent_name: str, ) -> None:`
- assignment provenance: rule
- reads state: `_model_name`, `logger`
- callers: `Terminus2._run_subagent`

## L3 `Terminus2._save_subagent_trajectory` — `terminus_2.py:1735-1800`
- signature: `def _save_subagent_trajectory( self, session_id: str, agent_name: str, steps: list[Step], usage_info, filename_suffix: str, summary_text: str, ) -> SubagentTraj`
- assignment provenance: rule-ambiguous(stage-8,stage-9)
- reads state: `_model_name`, `_session_id`, `_summarization_count`, `logger`, `logs_dir`
- callers: `Terminus2._run_subagent`
- calls: `Terminus2._extract_usage_metrics`, `Terminus2.version`

## L3 `Terminus2._convert_chat_messages_to_steps` — `terminus_2.py:1802-1860`
- signature: `def _convert_chat_messages_to_steps( self, chat_messages: list[dict[str, Any]], additional_user_message: str | None = None, mark_as_copied: bool = False, ) -> l`
- assignment provenance: rule
- reads state: `_last_response_model_name`, `_model_name`
- callers: `Terminus2._split_trajectory_on_summarization`

## L3 `Terminus2._dump_trajectory_with_continuation_index` — `terminus_2.py:1889-1955`
- signature: `def _dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:`
- assignment provenance: rule
- reads state: `_context`, `_linear_history`, `_llm_kwargs`, `_model_name`, `_parser_name`, `_session_id`, `_summarization_count`, `_temperature`, `_trajectory_steps`, `logger`, `logs_dir`
- callers: `Terminus2._dump_trajectory`, `Terminus2._split_trajectory_on_summarization`
- calls: `Terminus2.name`, `Terminus2.version`

## L3 `Terminus2._dump_trajectory` — `terminus_2.py:1957-1959`
- signature: `def _dump_trajectory(self) -> None:`
- assignment provenance: rule
- reads state: `_summarization_count`
- callers: `Terminus2._run_agent_loop`, `Terminus2.run`
- calls: `Terminus2._dump_trajectory_with_continuation_index`
