# stage-6 Command Execution (Terminal I/O)

Sending keystrokes to tmux, capturing panes, key batching, shell-session lifecycle.


## L3 `TmuxSession.__init__` — `tmux_session.py:51-80`
- signature: `def __init__( self, session_name: str, environment: BaseEnvironment, logging_path: Path | PurePosixPath, local_asciinema_recording_path: Path | None, remote_asc`
- assignment provenance: rule
- reads state: `_pane_height`, `_pane_width`
- writes state: `_disable_recording`, `_extra_env`, `_local_asciinema_recording_path`, `_logger`, `_logging_path`, `_markers`, `_pane_height`, `_pane_width`, `_previous_buffer`, `_remote_asciinema_recording_path`, `_session_name`, `_user`, `environment`
- callers: `Terminus2.setup`

## L3 `TmuxSession._attempt_tmux_installation` — `tmux_session.py:99-120`
- signature: `async def _attempt_tmux_installation(self) -> None:`
- assignment provenance: rule
- reads state: `_TOOL_INSTALL_BUDGET_SEC`, `_logger`
- callers: `TmuxSession.start`
- calls: `TmuxSession._install_recording_tools`

## L3 `TmuxSession._install_recording_tools` — `tmux_session.py:122-199`
- signature: `async def _install_recording_tools(self) -> None:`
- assignment provenance: rule-ambiguous(stage-6,stage-8)
- reads state: `_TOOL_INSTALL_TIMEOUT_SEC`, `_logger`, `_remote_asciinema_recording_path`, `environment`
- callers: `TmuxSession._attempt_tmux_installation`
- calls: `TmuxSession._build_tmux_from_source`, `TmuxSession._detect_system_info`, `TmuxSession._get_combined_install_command`, `TmuxSession._install_asciinema_with_pip`

## L3 `TmuxSession._detect_system_info` — `tmux_session.py:201-263`
- signature: `async def _detect_system_info(self) -> dict[str, str | None]:`
- assignment provenance: rule
- reads state: `_logger`, `environment`
- callers: `TmuxSession._install_recording_tools`

## L3 `TmuxSession._get_combined_install_command` — `tmux_session.py:265-291`
- signature: `def _get_combined_install_command( self, system_info: dict[str, Any], tools: list[str] ) -> str:`
- assignment provenance: rule
- callers: `TmuxSession._install_recording_tools`

## L3 `TmuxSession._build_tmux_from_source` — `tmux_session.py:293-343`
- signature: `async def _build_tmux_from_source(self) -> None:`
- assignment provenance: rule
- reads state: `_TOOL_INSTALL_TIMEOUT_SEC`, `_logger`, `environment`
- callers: `TmuxSession._install_recording_tools`

## L3 `TmuxSession._install_asciinema_with_pip` — `tmux_session.py:345-389`
- signature: `async def _install_asciinema_with_pip(self) -> None:`
- assignment provenance: rule
- reads state: `_TOOL_INSTALL_TIMEOUT_SEC`, `_logger`, `environment`
- callers: `TmuxSession._install_recording_tools`

## L3 `TmuxSession._tmux_start_session` — `tmux_session.py:392-405`
- signature: `def _tmux_start_session(self) -> str:`
- decorators: `property`
- assignment provenance: rule
- reads state: `_extra_env`, `_logging_path`, `_pane_height`, `_pane_width`, `_session_name`

## L3 `TmuxSession._utf8_len` — `tmux_session.py:408-414`
- signature: `def _utf8_len(s: str) -> int:`
- decorators: `staticmethod`
- assignment provenance: rule
- callers: `TmuxSession._batch_keys_for_send`, `TmuxSession._key_requires_paste`, `TmuxSession._max_escaped_key_length`

## L3 `TmuxSession._send_keys_prefix` — `tmux_session.py:417-420`
- signature: `def _send_keys_prefix(self) -> str:`
- decorators: `property`
- assignment provenance: rule
- reads state: `_session_name`

## L3 `TmuxSession._max_escaped_key_length` — `tmux_session.py:423-429`
- signature: `def _max_escaped_key_length(self) -> int:`
- decorators: `property`
- assignment provenance: rule
- reads state: `_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH`
- calls: `TmuxSession._utf8_len`

## L3 `TmuxSession._send_keys_command` — `tmux_session.py:431-432`
- signature: `def _send_keys_command(self, keys: list[str]) -> str:`
- assignment provenance: rule
- callers: `TmuxSession._send_key_batches`, `TmuxSession._send_single_key`, `TmuxSession._tmux_send_keys`

## L3 `TmuxSession._key_requires_paste` — `tmux_session.py:434-435`
- signature: `def _key_requires_paste(self, key: str) -> bool:`
- assignment provenance: rule
- callers: `TmuxSession._send_keys_to_session`
- calls: `TmuxSession._utf8_len`

## L3 `TmuxSession._batch_keys_for_send` — `tmux_session.py:437-467`
- signature: `def _batch_keys_for_send(self, keys: list[str]) -> list[list[str]]:`
- assignment provenance: rule
- reads state: `_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH`
- callers: `TmuxSession._send_key_batches`, `TmuxSession._tmux_send_keys`
- calls: `TmuxSession._utf8_len`

## L3 `TmuxSession._tmux_send_keys` — `tmux_session.py:469-478`
- signature: `def _tmux_send_keys(self, keys: list[str]) -> list[str]:`
- assignment provenance: rule
- calls: `TmuxSession._batch_keys_for_send`, `TmuxSession._send_keys_command`

## L3 `TmuxSession._tmux_capture_pane` — `tmux_session.py:480-495`
- signature: `def _tmux_capture_pane(self, capture_entire: bool = False) -> str:`
- assignment provenance: rule
- reads state: `_session_name`
- callers: `TmuxSession.capture_pane`

## L3 `TmuxSession.start` — `tmux_session.py:497-538`
- signature: `async def start(self) -> None:`
- assignment provenance: rule
- reads state: `GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH`, `_GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH`, `_logger`, `_remote_asciinema_recording_path`, `_user`, `environment`
- callers: `Terminus2.setup`
- calls: `TmuxSession._attempt_tmux_installation`, `TmuxSession.send_keys`

## L3 `TmuxSession.stop` — `tmux_session.py:540-577`
- signature: `async def stop(self) -> None:`
- assignment provenance: rule
- reads state: `_local_asciinema_recording_path`, `_logger`, `_markers`, `_remote_asciinema_recording_path`, `environment`
- calls: `AsciinemaHandler.__init__`, `TmuxSession.send_keys`

## L3 `TmuxSession._is_enter_key` — `tmux_session.py:579-580`
- signature: `def _is_enter_key(self, key: str) -> bool:`
- assignment provenance: rule
- reads state: `_ENTER_KEYS`
- callers: `TmuxSession._is_executing_command`, `TmuxSession._prevent_execution`

## L3 `TmuxSession._ends_with_newline` — `tmux_session.py:582-584`
- signature: `def _ends_with_newline(self, key: str) -> bool:`
- assignment provenance: rule
- reads state: `_ENDS_WITH_NEWLINE_PATTERN`
- callers: `TmuxSession._is_executing_command`

## L3 `TmuxSession.is_session_alive` — `tmux_session.py:586-592`
- signature: `async def is_session_alive(self) -> bool:`
- assignment provenance: rule
- reads state: `_session_name`, `_user`, `environment`
- callers: `Terminus2._run_agent_loop`

## L3 `TmuxSession._is_executing_command` — `tmux_session.py:594-595`
- signature: `def _is_executing_command(self, key: str) -> bool:`
- assignment provenance: rule
- callers: `TmuxSession._prepare_keys`, `TmuxSession._prevent_execution`
- calls: `TmuxSession._ends_with_newline`, `TmuxSession._is_enter_key`

## L3 `TmuxSession._prevent_execution` — `tmux_session.py:597-610`
- signature: `def _prevent_execution(self, keys: list[str]) -> list[str]:`
- assignment provenance: rule
- reads state: `_NEWLINE_CHARS`
- callers: `TmuxSession._prepare_keys`
- calls: `TmuxSession._is_enter_key`, `TmuxSession._is_executing_command`

## L3 `TmuxSession._prepare_keys` — `tmux_session.py:612-637`
- signature: `def _prepare_keys( self, keys: str | list[str], block: bool, ) -> tuple[list[str], bool]:`
- assignment provenance: rule
- reads state: `_TMUX_COMPLETION_COMMAND`
- callers: `TmuxSession.send_keys`
- calls: `TmuxSession._is_executing_command`, `TmuxSession._prevent_execution`

## L3 `TmuxSession._send_keys_error` — `tmux_session.py:639-646`
- signature: `def _send_keys_error( self, action: str, command: str, result: ExecResult ) -> RuntimeError:`
- assignment provenance: rule
- reads state: `environment`
- callers: `TmuxSession._paste_key`, `TmuxSession._send_key_batches`, `TmuxSession._send_single_key`

## L3 `TmuxSession._is_command_too_long_error` — `tmux_session.py:648-651`
- signature: `def _is_command_too_long_error(self, result: ExecResult) -> bool:`
- assignment provenance: rule
- reads state: `_TMUX_COMMAND_TOO_LONG_MARKER`
- callers: `TmuxSession._send_key_batches`, `TmuxSession._send_single_key`

## L3 `TmuxSession._paste_key` — `tmux_session.py:653-691`
- signature: `async def _paste_key(self, key: str, action: str) -> None:`
- assignment provenance: rule
- reads state: `_PASTE_BASE64_CHUNK_LEN`, `_session_name`, `_user`, `environment`
- callers: `TmuxSession._send_keys_to_session`, `TmuxSession._send_single_key`
- calls: `TmuxSession._send_keys_error`

## L3 `TmuxSession._send_single_key` — `tmux_session.py:693-703`
- signature: `async def _send_single_key(self, key: str, action: str) -> None:`
- assignment provenance: rule
- reads state: `_user`, `environment`
- callers: `TmuxSession._send_key_batches`
- calls: `TmuxSession._is_command_too_long_error`, `TmuxSession._paste_key`, `TmuxSession._send_keys_command`, `TmuxSession._send_keys_error`

## L3 `TmuxSession._send_key_batches` — `tmux_session.py:705-722`
- signature: `async def _send_key_batches(self, keys: list[str], action: str) -> None:`
- assignment provenance: rule
- reads state: `_logger`, `_user`, `environment`
- callers: `TmuxSession._send_keys_to_session`
- calls: `TmuxSession._batch_keys_for_send`, `TmuxSession._is_command_too_long_error`, `TmuxSession._send_keys_command`, `TmuxSession._send_keys_error`, `TmuxSession._send_single_key`

## L3 `TmuxSession._send_keys_to_session` — `tmux_session.py:724-745`
- signature: `async def _send_keys_to_session(self, keys: list[str], action: str) -> None:`
- assignment provenance: rule
- callers: `TmuxSession._send_blocking_keys`, `TmuxSession._send_non_blocking_keys`
- calls: `TmuxSession._key_requires_paste`, `TmuxSession._paste_key`, `TmuxSession._send_key_batches`

## L3 `TmuxSession._send_blocking_keys` — `tmux_session.py:747-763`
- signature: `async def _send_blocking_keys( self, keys: list[str], max_timeout_sec: float, ):`
- assignment provenance: rule
- reads state: `_logger`, `_user`, `environment`
- callers: `TmuxSession.send_keys`
- calls: `TmuxSession._send_keys_to_session`

## L3 `TmuxSession._send_non_blocking_keys` — `tmux_session.py:765-777`
- signature: `async def _send_non_blocking_keys( self, keys: list[str], min_timeout_sec: float, ):`
- assignment provenance: rule
- callers: `TmuxSession.send_keys`
- calls: `TmuxSession._send_keys_to_session`

## L3 `TmuxSession.send_keys` — `tmux_session.py:779-820`
- signature: `async def send_keys( self, keys: str | list[str], block: bool = False, min_timeout_sec: float = 0.0, max_timeout_sec: float = 180.0, ):`
- assignment provenance: rule
- reads state: `_logger`
- callers: `Terminus2._execute_commands`, `TmuxSession.start`, `TmuxSession.stop`
- calls: `TmuxSession._prepare_keys`, `TmuxSession._send_blocking_keys`, `TmuxSession._send_non_blocking_keys`

## L3 `TmuxSession.capture_pane` — `tmux_session.py:822-826`
- signature: `async def capture_pane(self, capture_entire: bool = False) -> str:`
- assignment provenance: rule
- reads state: `_user`, `environment`
- callers: `Terminus2._summarize`, `TmuxSession._get_visible_screen`, `TmuxSession.get_incremental_output`
- calls: `TmuxSession._tmux_capture_pane`

## L3 `TmuxSession._get_visible_screen` — `tmux_session.py:828-829`
- signature: `async def _get_visible_screen(self) -> str:`
- assignment provenance: rule
- callers: `TmuxSession.get_incremental_output`
- calls: `TmuxSession.capture_pane`

## L3 `TmuxSession._find_new_content` — `tmux_session.py:831-839`
- signature: `async def _find_new_content(self, current_buffer: str) -> str | None:`
- assignment provenance: rule
- reads state: `_previous_buffer`
- callers: `TmuxSession.get_incremental_output`

## L3 `TmuxSession.get_incremental_output` — `tmux_session.py:841-876`
- signature: `async def get_incremental_output(self) -> str:`
- assignment provenance: rule
- reads state: `_previous_buffer`
- writes state: `_previous_buffer`
- callers: `Terminus2._execute_commands`, `Terminus2.run`
- calls: `TmuxSession._find_new_content`, `TmuxSession._get_visible_screen`, `TmuxSession.capture_pane`
