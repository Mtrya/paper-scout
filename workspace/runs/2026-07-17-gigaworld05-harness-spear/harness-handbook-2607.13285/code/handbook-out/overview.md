# L1 System Overview — Terminus-2 (probe-generated)

Terminus-2 is a terminal agent in an observe-decide-act loop: it captures the tmux pane, prompts an LLM, parses the response into commands, sends keystrokes to the terminal, and repeats until the model marks the task complete (twice, via a pending-completion handshake) or limits are hit. Context pressure is handled by proactive summarization; everything is recorded as trajectories.

Stages: `Initialization & Config`, `Prompt & Template Assembly`, `LLM Query & API Handling`, `Response Parsing`, `Main Agent Loop`, `Command Execution (Terminal I/O)`, `Context & Token Management`, `Trajectory & Recording`, `Subagent Handoff`, `Per-Run Lifecycle & Reset`
