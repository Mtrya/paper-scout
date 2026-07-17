# L2 Index — stages and leaves


## stage-1 Initialization & Config
Agent construction, LLM client setup, model-info resolution, API surface (name/version).

- `Terminus2._init_llm` — terminus_2.py:82-154 — Initialize the LLM backend based on llm_backend parameter.
- `Terminus2.__init__` — terminus_2.py:156-344 — Initialize Terminus 2 agent.
- `Terminus2._resolve_model_info` — terminus_2.py:346-358 — (no docstring)
- `Terminus2.name` — terminus_2.py:362-363 — (no docstring)
- `Terminus2.version` — terminus_2.py:366-367 — (no docstring)

## stage-2 Prompt & Template Assembly
Loading prompt/timeout templates, skill frontmatter, instruction construction.

- `Terminus2._get_prompt_template_path` — terminus_2.py:402-411 — Return the path to the prompt template for this format.
- `Terminus2._get_timeout_template_path` — terminus_2.py:413-415 — Return the path to the timeout template for this format.
- `Terminus2._parse_skill_frontmatter` — terminus_2.py:418-433 — Parse YAML frontmatter from SKILL.md content, returning name and description.

## stage-3 LLM Query & API Handling
Sending chat to the model, retries, error classification, usage/latency metrics.

- `Terminus2._get_error_response_type` — terminus_2.py:488-500 — Return the response type name for error messages.
- `Terminus2._get_completion_confirmation_message` — terminus_2.py:502-523 — Return the format-specific task completion confirmation message.
- `Terminus2._extract_usage_metrics` — terminus_2.py:633-650 — Extract and normalize metrics from usage info.
- `Terminus2._track_api_request_time` — terminus_2.py:652-660 — Track API request time from start timestamp.
- `Terminus2._query_llm` — terminus_2.py:995-1155 — (no docstring)

## stage-4 Response Parsing
Turning raw model text into commands + completion flags (JSON/XML parsers).

- `Terminus2._get_parser` — terminus_2.py:391-400 — Return the appropriate parser instance for this format.
- `TerminusJSONPlainParser.__init__` — terminus_json_plain_parser.py:26-27 — (no docstring)
- `TerminusJSONPlainParser.parse_response` — terminus_json_plain_parser.py:29-62 — Parse a terminus JSON plain response and extract commands.
- `TerminusJSONPlainParser._try_parse_response` — terminus_json_plain_parser.py:64-163 — Try to parse a terminus JSON plain response.
- `TerminusJSONPlainParser._extract_json_content` — terminus_json_plain_parser.py:165-212 — Extract JSON content from response, handling extra text.
- `TerminusJSONPlainParser._validate_json_structure` — terminus_json_plain_parser.py:214-249 — Validate the JSON structure has required fields.
- `TerminusJSONPlainParser._parse_commands` — terminus_json_plain_parser.py:251-303 — Parse commands array into ParsedCommand objects.
- `TerminusJSONPlainParser._get_auto_fixes` — terminus_json_plain_parser.py:305-313 — Return list of auto-fix functions to try in order.
- `TerminusJSONPlainParser._fix_incomplete_json` — terminus_json_plain_parser.py:315-328 — Fix incomplete JSON by adding missing closing braces.
- `TerminusJSONPlainParser._fix_mixed_content` — terminus_json_plain_parser.py:330-343 — Extract JSON from response with mixed content.
- `TerminusJSONPlainParser._combine_warnings` — terminus_json_plain_parser.py:345-350 — Combine auto-correction warning with existing warnings.
- `TerminusJSONPlainParser._check_field_order` — terminus_json_plain_parser.py:352-393 — Check if fields appear in the correct order: analysis, plan, commands.
- `TerminusXMLPlainParser.__init__` — terminus_xml_plain_parser.py:25-26 — (no docstring)
- `TerminusXMLPlainParser.parse_response` — terminus_xml_plain_parser.py:28-60 — Parse a terminus XML plain response and extract commands.
- `TerminusXMLPlainParser._try_parse_response` — terminus_xml_plain_parser.py:62-169 — Try to parse a terminus XML plain response.
- `TerminusXMLPlainParser._get_auto_fixes` — terminus_xml_plain_parser.py:171-178 — Return list of auto-fix functions to try in order.
- `TerminusXMLPlainParser._combine_warnings` — terminus_xml_plain_parser.py:180-185 — Combine auto-correction warning with existing warnings.
- `TerminusXMLPlainParser._fix_missing_response_tag` — terminus_xml_plain_parser.py:187-194 — Fix missing </response> closing tag by appending it.
- `TerminusXMLPlainParser._check_extra_text` — terminus_xml_plain_parser.py:196-223 — Check for extra text before/after <response> tags.
- `TerminusXMLPlainParser._extract_response_content` — terminus_xml_plain_parser.py:225-236 — Extract content from <response> tags.
- `TerminusXMLPlainParser._extract_sections` — terminus_xml_plain_parser.py:238-318 — Extract analysis, plan, commands, and task_complete sections.
- `TerminusXMLPlainParser._parse_xml_commands` — terminus_xml_plain_parser.py:320-391 — Parse XML content and extract command objects manually.
- `TerminusXMLPlainParser._find_top_level_tags` — terminus_xml_plain_parser.py:393-440 — Find all top-level XML tags (direct children of response), not
- `TerminusXMLPlainParser._check_section_order` — terminus_xml_plain_parser.py:442-480 — Check if sections appear in the correct order: analysis, plan, commands.
- `TerminusXMLPlainParser._check_attribute_issues` — terminus_xml_plain_parser.py:482-512 — Check for attribute-related issues.
- `TerminusXMLPlainParser._check_task_complete` — terminus_xml_plain_parser.py:514-526 — Check if the response indicates the task is complete.
- `TerminusXMLPlainParser.salvage_truncated_response` — terminus_xml_plain_parser.py:528-580 — Try to salvage a valid response from truncated output.

## stage-5 Main Agent Loop
The observe-decide-act iteration: episode bookkeeping, observation handling, command dispatch, and the completion gate that decides loop termination.

- `_terminal_observation_source_call_id` — terminus_2.py:61-66 — (no docstring)
- `Terminus2._handle_llm_interaction` — terminus_2.py:1157-1197 — (no docstring)
- `Terminus2._execute_commands` — terminus_2.py:1199-1229 — Execute a batch of commands in the terminal.
- `Terminus2._run_agent_loop` — terminus_2.py:1231-1540 — (no docstring)

## stage-6 Command Execution (Terminal I/O)
Sending keystrokes to tmux, capturing panes, key batching, shell-session lifecycle.

- `TmuxSession.__init__` — tmux_session.py:51-80 — (no docstring)
- `TmuxSession._attempt_tmux_installation` — tmux_session.py:99-120 — Install tmux and asciinema, bounded so a hung install cannot consume
- `TmuxSession._install_recording_tools` — tmux_session.py:122-199 — Install both tmux and asciinema in a single operation for efficiency.
- `TmuxSession._detect_system_info` — tmux_session.py:201-263 — Detect the operating system and available package managers.
- `TmuxSession._get_combined_install_command` — tmux_session.py:265-291 — Get the appropriate installation command for multiple tools based on system info.
- `TmuxSession._build_tmux_from_source` — tmux_session.py:293-343 — Build tmux from source as a fallback option.
- `TmuxSession._install_asciinema_with_pip` — tmux_session.py:345-389 — Install asciinema using pip as a fallback.
- `TmuxSession._tmux_start_session` — tmux_session.py:392-405 — (no docstring)
- `TmuxSession._utf8_len` — tmux_session.py:408-414 — Return the UTF-8 byte length of *s*.
- `TmuxSession._send_keys_prefix` — tmux_session.py:417-420 — (no docstring)
- `TmuxSession._max_escaped_key_length` — tmux_session.py:423-429 — Largest quoted key that still fits in a single send-keys command.
- `TmuxSession._send_keys_command` — tmux_session.py:431-432 — (no docstring)
- `TmuxSession._key_requires_paste` — tmux_session.py:434-435 — (no docstring)
- `TmuxSession._batch_keys_for_send` — tmux_session.py:437-467 — Group *keys* into batches whose send-keys command fits the limit.
- `TmuxSession._tmux_send_keys` — tmux_session.py:469-478 — Build one or more ``tmux send-keys`` commands for *keys*.
- `TmuxSession._tmux_capture_pane` — tmux_session.py:480-495 — (no docstring)
- `TmuxSession.start` — tmux_session.py:497-538 — (no docstring)
- `TmuxSession.stop` — tmux_session.py:540-577 — (no docstring)
- `TmuxSession._is_enter_key` — tmux_session.py:579-580 — (no docstring)
- `TmuxSession._ends_with_newline` — tmux_session.py:582-584 — (no docstring)
- `TmuxSession.is_session_alive` — tmux_session.py:586-592 — Check if the tmux session is still alive.
- `TmuxSession._is_executing_command` — tmux_session.py:594-595 — (no docstring)
- `TmuxSession._prevent_execution` — tmux_session.py:597-610 — (no docstring)
- `TmuxSession._prepare_keys` — tmux_session.py:612-637 — Prepare keys for sending to the terminal.
- `TmuxSession._send_keys_error` — tmux_session.py:639-646 — (no docstring)
- `TmuxSession._is_command_too_long_error` — tmux_session.py:648-651 — (no docstring)
- `TmuxSession._paste_key` — tmux_session.py:653-691 — Deliver *key* to the pane via a tmux paste buffer.
- `TmuxSession._send_single_key` — tmux_session.py:693-703 — (no docstring)
- `TmuxSession._send_key_batches` — tmux_session.py:705-722 — (no docstring)
- `TmuxSession._send_keys_to_session` — tmux_session.py:724-745 — Send *keys* in order, pasting oversized literal keys.
- `TmuxSession._send_blocking_keys` — tmux_session.py:747-763 — (no docstring)
- `TmuxSession._send_non_blocking_keys` — tmux_session.py:765-777 — (no docstring)
- `TmuxSession.send_keys` — tmux_session.py:779-820 — Execute a command in the tmux session.
- `TmuxSession.capture_pane` — tmux_session.py:822-826 — (no docstring)
- `TmuxSession._get_visible_screen` — tmux_session.py:828-829 — (no docstring)
- `TmuxSession._find_new_content` — tmux_session.py:831-839 — (no docstring)
- `TmuxSession.get_incremental_output` — tmux_session.py:841-876 — Get either new terminal output since last call, or current screen if

## stage-7 Context & Token Management
Token counting, output capping, proactive summarization, message unwinding.

- `Terminus2._count_total_tokens` — terminus_2.py:525-529 — Count total tokens across all messages in the chat.
- `Terminus2._limit_output_length` — terminus_2.py:531-567 — Limit output to specified byte length, keeping first and last portions.
- `Terminus2._unwind_messages_to_free_tokens` — terminus_2.py:569-593 — Remove recent messages until we have enough free tokens.
- `Terminus2._summarize` — terminus_2.py:741-955 — Create a summary of the agent's work to pass to a new agent instance.
- `Terminus2._check_proactive_summarization` — terminus_2.py:957-981 — Check if we should proactively summarize due to token usage.
- `Terminus2._split_trajectory_on_summarization` — terminus_2.py:1862-1887 — Split trajectory on summarization when linear_history is enabled.

## stage-8 Trajectory & Recording
Step/trajectory persistence, asciinema recording, ATIF conversion.

- `AsciinemaHandler.__init__` — asciinema_handler.py:11-20 — Initialize the AsciinemaHandler.
- `AsciinemaHandler.merge_markers` — asciinema_handler.py:22-39 — Merge asciinema markers into a recording.
- `AsciinemaHandler._write_merged_recording` — asciinema_handler.py:41-60 — Write a new recording file with markers merged in at the correct timestamps.
- `AsciinemaHandler._process_recording_line` — asciinema_handler.py:62-90 — Process a single line from the recording, inserting markers as needed.
- `AsciinemaHandler._write_marker` — asciinema_handler.py:92-96 — Write a single marker event to the output file.
- `AsciinemaHandler._write_remaining_markers` — asciinema_handler.py:98-103 — Write any remaining markers that come after all recorded events.
- `Terminus2._run_subagent` — terminus_2.py:662-739 — Run a subagent and return its response and trajectory reference.
- `Terminus2._remove_metrics_from_copied_steps` — terminus_2.py:1648-1665 — Remove metrics from copied trajectory steps and mark as copied context.
- `Terminus2._prepare_copied_trajectory_steps` — terminus_2.py:1667-1681 — Prepare trajectory steps for subagent by copying and removing metrics.
- `Terminus2._append_subagent_response_step` — terminus_2.py:1683-1733 — Append a response step with conditional metrics to trajectory steps.
- `Terminus2._save_subagent_trajectory` — terminus_2.py:1735-1800 — Save a subagent trajectory to disk and return its reference.
- `Terminus2._convert_chat_messages_to_steps` — terminus_2.py:1802-1860 — Convert chat messages to trajectory steps.
- `Terminus2._dump_trajectory_with_continuation_index` — terminus_2.py:1889-1955 — Dump trajectory data to JSON file with specified continuation index.
- `Terminus2._dump_trajectory` — terminus_2.py:1957-1959 — Dump trajectory data to JSON file following ATIF format.

## stage-9 Subagent Handoff
Spawning subagents with summarized context, aggregating their metrics.

- `Terminus2._collect_subagent_rollout_detail` — terminus_2.py:595-617 — Collect rollout details from a subagent LLM response.
- `Terminus2._update_subagent_metrics` — terminus_2.py:619-631 — Update subagent metrics with usage information from an LLM response.

## stage-10 Per-Run Lifecycle & Reset
run()/perform_task entry points and per-run state reset.

- `Terminus2._build_skills_section` — terminus_2.py:435-486 — Discover Agent Skills in skills_dir and return an <available_skills> XML block.
- `Terminus2._reset_per_run_state` — terminus_2.py:1542-1556 — Reset all per-run state. The same Terminus2 instance is reused
- `Terminus2.run` — terminus_2.py:1559-1643 — (no docstring)

## UNMAPPED (explicit coverage record)

- `Terminus2.setup` — terminus_2.py:370-389
