# State registers — every self._* attribute with its read/write sites


## `GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH`
- writes: —
- reads: `TmuxSession.start` (tmux_session.py:497-538)

## `_ENDS_WITH_NEWLINE_PATTERN`
- writes: —
- reads: `TmuxSession._ends_with_newline` (tmux_session.py:582-584)

## `_ENTER_KEYS`
- writes: —
- reads: `TmuxSession._is_enter_key` (tmux_session.py:579-580)

## `_GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH`
- writes: —
- reads: `TmuxSession.start` (tmux_session.py:497-538)

## `_NEWLINE_CHARS`
- writes: —
- reads: `TmuxSession._prevent_execution` (tmux_session.py:597-610)

## `_PASTE_BASE64_CHUNK_LEN`
- writes: —
- reads: `TmuxSession._paste_key` (tmux_session.py:653-691)

## `_TMUX_COMMAND_TOO_LONG_MARKER`
- writes: —
- reads: `TmuxSession._is_command_too_long_error` (tmux_session.py:648-651)

## `_TMUX_COMPLETION_COMMAND`
- writes: —
- reads: `TmuxSession._prepare_keys` (tmux_session.py:612-637)

## `_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH`
- writes: —
- reads: `TmuxSession._batch_keys_for_send` (tmux_session.py:437-467), `TmuxSession._max_escaped_key_length` (tmux_session.py:423-429)

## `_TOOL_INSTALL_BUDGET_SEC`
- writes: —
- reads: `TmuxSession._attempt_tmux_installation` (tmux_session.py:99-120)

## `_TOOL_INSTALL_TIMEOUT_SEC`
- writes: —
- reads: `TmuxSession._build_tmux_from_source` (tmux_session.py:293-343), `TmuxSession._install_asciinema_with_pip` (tmux_session.py:345-389), `TmuxSession._install_recording_tools` (tmux_session.py:122-199)

## `_api_request_times`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556)
- reads: `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._track_api_request_time` (terminus_2.py:652-660), `Terminus2.run` (terminus_2.py:1559-1643)

## `_chat`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2.run` (terminus_2.py:1559-1643)
- reads: `Terminus2._split_trajectory_on_summarization` (terminus_2.py:1862-1887), `Terminus2.run` (terminus_2.py:1559-1643)

## `_collect_rollout_details`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._collect_subagent_rollout_detail` (terminus_2.py:595-617)

## `_context`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2.run` (terminus_2.py:1559-1643)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_disable_recording`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: —

## `_enable_summarize`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_extra_env`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `Terminus2.setup` (terminus_2.py:370-389), `TmuxSession._tmux_start_session` (tmux_session.py:392-405)

## `_interleaved_thinking`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.run` (terminus_2.py:1559-1643)

## `_last_response_model_name`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)
- reads: `Terminus2._convert_chat_messages_to_steps` (terminus_2.py:1802-1860)

## `_linear_history`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_llm`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._check_proactive_summarization` (terminus_2.py:957-981), `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._run_subagent` (terminus_2.py:662-739), `Terminus2._unwind_messages_to_free_tokens` (terminus_2.py:569-593), `Terminus2.run` (terminus_2.py:1559-1643)

## `_llm_call_kwargs`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._run_subagent` (terminus_2.py:662-739)

## `_llm_kwargs`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955)

## `_local_asciinema_recording_path`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession.stop` (tmux_session.py:540-577)

## `_logger`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._attempt_tmux_installation` (tmux_session.py:99-120), `TmuxSession._build_tmux_from_source` (tmux_session.py:293-343), `TmuxSession._detect_system_info` (tmux_session.py:201-263), `TmuxSession._install_asciinema_with_pip` (tmux_session.py:345-389), `TmuxSession._install_recording_tools` (tmux_session.py:122-199), `TmuxSession._send_blocking_keys` (tmux_session.py:747-763), `TmuxSession._send_key_batches` (tmux_session.py:705-722), `TmuxSession.send_keys` (tmux_session.py:779-820), `TmuxSession.start` (tmux_session.py:497-538), `TmuxSession.stop` (tmux_session.py:540-577)

## `_logging_path`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._tmux_start_session` (tmux_session.py:392-405)

## `_markers`
- writes: `AsciinemaHandler.__init__` (asciinema_handler.py:11-20), `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `AsciinemaHandler._process_recording_line` (asciinema_handler.py:62-90), `AsciinemaHandler._write_merged_recording` (asciinema_handler.py:41-60), `AsciinemaHandler.merge_markers` (asciinema_handler.py:22-39), `TmuxSession.stop` (tmux_session.py:540-577)

## `_max_episodes`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_model_name`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._append_subagent_response_step` (terminus_2.py:1683-1733), `Terminus2._convert_chat_messages_to_steps` (terminus_2.py:1802-1860), `Terminus2._count_total_tokens` (terminus_2.py:525-529), `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540), `Terminus2._save_subagent_trajectory` (terminus_2.py:1735-1800), `Terminus2._summarize` (terminus_2.py:741-955)

## `_n_episodes`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)
- reads: `Terminus2.run` (terminus_2.py:1559-1643)

## `_pane_height`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession.__init__` (tmux_session.py:51-80), `TmuxSession._tmux_start_session` (tmux_session.py:392-405)

## `_pane_width`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession.__init__` (tmux_session.py:51-80), `TmuxSession._tmux_start_session` (tmux_session.py:392-405)

## `_parser`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._handle_llm_interaction` (terminus_2.py:1157-1197), `Terminus2._query_llm` (terminus_2.py:995-1155)

## `_parser_name`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._get_completion_confirmation_message` (terminus_2.py:502-523), `Terminus2._get_error_response_type` (terminus_2.py:488-500), `Terminus2._get_parser` (terminus_2.py:391-400), `Terminus2._get_prompt_template_path` (terminus_2.py:402-411)

## `_pending_completion`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_pending_handoff_prompt`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_pending_subagent_refs`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_previous_buffer`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80), `TmuxSession.get_incremental_output` (tmux_session.py:841-876)
- reads: `TmuxSession._find_new_content` (tmux_session.py:831-839), `TmuxSession.get_incremental_output` (tmux_session.py:841-876)

## `_proactive_summarization_threshold`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._check_proactive_summarization` (terminus_2.py:957-981)

## `_prompt_template`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.run` (terminus_2.py:1559-1643)

## `_reasoning_effort`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: —

## `_record_terminal_session`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.setup` (terminus_2.py:370-389)

## `_recording_path`
- writes: `AsciinemaHandler.__init__` (asciinema_handler.py:11-20)
- reads: `AsciinemaHandler._write_merged_recording` (asciinema_handler.py:41-60), `AsciinemaHandler.merge_markers` (asciinema_handler.py:22-39)

## `_remote_asciinema_recording_path`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._install_recording_tools` (tmux_session.py:122-199), `TmuxSession.start` (tmux_session.py:497-538), `TmuxSession.stop` (tmux_session.py:540-577)

## `_save_raw_content_in_trajectory`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540)

## `_session`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2.setup` (terminus_2.py:370-389)
- reads: `Terminus2._run_agent_loop` (terminus_2.py:1231-1540), `Terminus2.run` (terminus_2.py:1559-1643), `Terminus2.setup` (terminus_2.py:370-389)

## `_session_id`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._split_trajectory_on_summarization` (terminus_2.py:1862-1887)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._save_subagent_trajectory` (terminus_2.py:1735-1800), `Terminus2._split_trajectory_on_summarization` (terminus_2.py:1862-1887), `Terminus2._summarize` (terminus_2.py:741-955)

## `_session_name`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._paste_key` (tmux_session.py:653-691), `TmuxSession._send_keys_prefix` (tmux_session.py:417-420), `TmuxSession._tmux_capture_pane` (tmux_session.py:480-495), `TmuxSession._tmux_start_session` (tmux_session.py:392-405), `TmuxSession.is_session_alive` (tmux_session.py:586-592)

## `_store_all_messages`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.run` (terminus_2.py:1559-1643)

## `_subagent_metrics`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556)
- reads: `Terminus2._update_subagent_metrics` (terminus_2.py:619-631), `Terminus2.run` (terminus_2.py:1559-1643)

## `_subagent_rollout_details`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556)
- reads: `Terminus2._collect_subagent_rollout_detail` (terminus_2.py:595-617), `Terminus2.run` (terminus_2.py:1559-1643)

## `_summarization_count`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._summarize` (terminus_2.py:741-955)
- reads: `Terminus2._dump_trajectory` (terminus_2.py:1957-1959), `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._save_subagent_trajectory` (terminus_2.py:1735-1800), `Terminus2._split_trajectory_on_summarization` (terminus_2.py:1862-1887), `Terminus2._summarize` (terminus_2.py:741-955), `Terminus2.run` (terminus_2.py:1559-1643)

## `_temperature`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955)

## `_timeout_template`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._execute_commands` (terminus_2.py:1199-1229)

## `_tmux_pane_height`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.setup` (terminus_2.py:370-389)

## `_tmux_pane_width`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.setup` (terminus_2.py:370-389)

## `_trajectory_config`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2.__init__` (terminus_2.py:156-344)

## `_trajectory_steps`
- writes: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556), `Terminus2._split_trajectory_on_summarization` (terminus_2.py:1862-1887)
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._prepare_copied_trajectory_steps` (terminus_2.py:1667-1681), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540), `Terminus2.run` (terminus_2.py:1559-1643)

## `_user`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._paste_key` (tmux_session.py:653-691), `TmuxSession._send_blocking_keys` (tmux_session.py:747-763), `TmuxSession._send_key_batches` (tmux_session.py:705-722), `TmuxSession._send_single_key` (tmux_session.py:693-703), `TmuxSession.capture_pane` (tmux_session.py:822-826), `TmuxSession.is_session_alive` (tmux_session.py:586-592), `TmuxSession.start` (tmux_session.py:497-538)

## `_user_provided_session_id`
- writes: `Terminus2.__init__` (terminus_2.py:156-344)
- reads: `Terminus2._reset_per_run_state` (terminus_2.py:1542-1556)

## `environment`
- writes: `TmuxSession.__init__` (tmux_session.py:51-80)
- reads: `TmuxSession._build_tmux_from_source` (tmux_session.py:293-343), `TmuxSession._detect_system_info` (tmux_session.py:201-263), `TmuxSession._install_asciinema_with_pip` (tmux_session.py:345-389), `TmuxSession._install_recording_tools` (tmux_session.py:122-199), `TmuxSession._paste_key` (tmux_session.py:653-691), `TmuxSession._send_blocking_keys` (tmux_session.py:747-763), `TmuxSession._send_key_batches` (tmux_session.py:705-722), `TmuxSession._send_keys_error` (tmux_session.py:639-646), `TmuxSession._send_single_key` (tmux_session.py:693-703), `TmuxSession.capture_pane` (tmux_session.py:822-826), `TmuxSession.is_session_alive` (tmux_session.py:586-592), `TmuxSession.start` (tmux_session.py:497-538), `TmuxSession.stop` (tmux_session.py:540-577)

## `logger`
- writes: —
- reads: `Terminus2.__init__` (terminus_2.py:156-344), `Terminus2._append_subagent_response_step` (terminus_2.py:1683-1733), `Terminus2._check_proactive_summarization` (terminus_2.py:957-981), `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._handle_llm_interaction` (terminus_2.py:1157-1197), `Terminus2._query_llm` (terminus_2.py:995-1155), `Terminus2._resolve_model_info` (terminus_2.py:346-358), `Terminus2._run_agent_loop` (terminus_2.py:1231-1540), `Terminus2._save_subagent_trajectory` (terminus_2.py:1735-1800), `Terminus2._unwind_messages_to_free_tokens` (terminus_2.py:569-593)

## `logs_dir`
- writes: —
- reads: `Terminus2._dump_trajectory_with_continuation_index` (terminus_2.py:1889-1955), `Terminus2._save_subagent_trajectory` (terminus_2.py:1735-1800)

## `mcp_servers`
- writes: —
- reads: `Terminus2.run` (terminus_2.py:1559-1643)

## `required_fields`
- writes: `TerminusJSONPlainParser.__init__` (terminus_json_plain_parser.py:26-27)
- reads: `TerminusJSONPlainParser._validate_json_structure` (terminus_json_plain_parser.py:214-249)

## `required_sections`
- writes: `TerminusXMLPlainParser.__init__` (terminus_xml_plain_parser.py:25-26)
- reads: `TerminusXMLPlainParser._extract_sections` (terminus_xml_plain_parser.py:238-318)

## `skills_dir`
- writes: —
- reads: `Terminus2._build_skills_section` (terminus_2.py:435-486)
