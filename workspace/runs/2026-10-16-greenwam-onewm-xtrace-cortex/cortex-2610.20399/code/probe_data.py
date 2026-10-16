"""Data for the Cortex prospective-query probe.

C = a fictional internal engineering document ("meridian").
Two held-out eval kinds:
  RESTATE : recall questions whose answer is stated verbatim in C.
  USE     : application questions that anticipate using C while doing a job.
Both kinds are scored as 4-way forced choice over candidate answer strings.
"""

CONTEXT = """\
meridian is an internal Python library for scheduling batch jobs on shared compute clusters.

Installation: pip install meridian-pipeline==3.4.1. It requires Python 3.11 or newer. The package is imported as `import meridian`.

Configuration: meridian reads exactly one configuration file, meridian.toml, from the working directory. It never reads environment variables. The keys are:
- queue: name of the job queue. Default is "alpha".
- retries: number of automatic retry attempts per job. Default is 2. Allowed range is 0 to 9.
- timeout_s: per-job timeout in seconds. Default is 900.
- backpressure: behaviour when the queue is full. One of "drop", "block", "spill". Default is "block".

Behaviour rules:
- With backpressure set to "spill", overflow jobs are written to /var/meridian/overflow and resumed on the next scheduler start.
- Setting retries to 0 disables the automatic retry logic entirely.
- The scheduler applies a configuration change only after a restart. There is no hot reload.
- Every job must declare a resources field. A job without it is rejected at submission with error code M412.

Diagnostics:
- `meridian doctor` prints the resolved configuration and exits.
- `meridian trace <job_id>` prints the last 50 state transitions of that job.
- Error M412 means a job is missing its resources field. Error M517 means the configuration file was not found.
- Metrics are exposed at /metrics on port 9633.
"""

# ---------------------------------------------------------------- eval: restate
# (question, correct answer, distractors)  -- answer is candidate index 0
EVAL_RESTATE = [
    ("What is the default value of the queue key in meridian.toml?",
     "alpha", ["beta", "main", "default"]),
    ("How many automatic retry attempts does meridian use by default?",
     "2", ["3", "0", "5"]),
    ("What is the default per-job timeout in meridian?",
     "900 seconds", ["600 seconds", "1800 seconds", "300 seconds"]),
    ("What is the default backpressure behaviour of meridian?",
     "block", ["drop", "spill", "retry"]),
    ("Which configuration file does meridian read?",
     "meridian.toml", ["config.toml", "meridian.yaml", ".meridianrc"]),
    ("Which Python versions does meridian-pipeline 3.4.1 require?",
     "3.11 or newer", ["3.8 or newer", "3.9 or newer", "3.12 or newer"]),
    ("What does error code M517 mean?",
     "the configuration file was not found", ["a job is missing its resources field", "the queue is full", "the job timed out"]),
    ("On which port does meridian expose its metrics?",
     "9633", ["8080", "9090", "6339"]),
    ("How many state transitions does `meridian trace` print?",
     "the last 50", ["the last 10", "the last 100", "all of them"]),
    ("What is the pip package name for meridian?",
     "meridian-pipeline", ["meridian", "meridian-scheduler", "pymeridian"]),
    ("What is the allowed range of the retries key?",
     "0 to 9", ["0 to 5", "1 to 9", "0 to 3"]),
    ("Can meridian pick up a configuration change while it is running?",
     "no, it only applies the change after a restart", ["yes, it reloads meridian.toml automatically", "yes, but only for the retries key", "yes, if hot_reload is enabled in meridian.toml"]),
    ("What is the exact version string installed by `pip install meridian-pipeline==...`?",
     "3.4.1", ["3.4.0", "4.3.1", "3.14.1"]),
    ("How is meridian imported in Python code?",
     "import meridian", ["import meridian_pipeline", "from meridian import pipeline", "import meridians"]),
    ("Does meridian read configuration from environment variables?",
     "no, only from meridian.toml", ["yes, they override meridian.toml", "yes, MERIDIAN_QUEUE is required", "only for the queue key"]),
    ("What does the command `meridian doctor` print?",
     "the resolved configuration", ["the last 50 state transitions of a job", "the job queue contents", "the number of running workers"]),
    ("Which field must every meridian job declare?",
     "resources", ["queue", "timeout_s", "priority"]),
    ("What happens to a submitted job that has no resources field?",
     "it is rejected at submission with error M412", ["it is queued with default resources", "it is silently dropped", "it is accepted but never scheduled"]),
    ("Which value of backpressure makes meridian write overflow jobs to disk?",
     "spill", ["block", "drop", "buffered"]),
    ("What does error code M412 mean?",
     "a job is missing its resources field", ["the configuration file was not found", "the queue is full", "a job exceeded its timeout"]),
]

# ------------------------------------------------------------------- eval: use
EVAL_USE = [
    ("A job must survive a scheduler restart and continue later. Which backpressure setting achieves this?",
     "spill", ["drop", "block", "defer"]),
    ("You want a full queue to reject new submissions instead of delaying them. Which setting do you change?",
     "backpressure to \"drop\"", ["backpressure to \"block\"", "backpressure to \"spill\"", "retries to 0"]),
    ("You edited meridian.toml but the running scheduler still uses the old values. What must you do?",
     "restart the scheduler", ["wait one minute for the reload", "run meridian reload", "nothing, it applies the change automatically"]),
    ("A job is rejected at submission with M412. What is wrong with it?",
     "it has no resources field", ["the queue is full", "the configuration file is missing", "its timeout is too small"]),
    ("You need to know which configuration values meridian actually resolved at runtime. Which command do you run?",
     "meridian doctor", ["meridian trace", "meridian status", "meridian config"]),
    ("A job keeps restarting and you want to inspect its recent state transitions. Which command do you run?",
     "meridian trace <job_id>", ["meridian doctor <job_id>", "meridian restart <job_id>", "meridian logs <job_id>"]),
    ("You want a failing job to be attempted exactly once, with no automatic retries. What do you set?",
     "retries = 0", ["retries = 1", "retries = 2", "timeout_s = 0"]),
    ("Jobs are being killed after ten minutes but you expect 20-minute runs. Which key do you change?",
     "timeout_s", ["retries", "queue", "backpressure"]),
    ("You need meridian to submit into the beta queue. Where do you change that in this setup?",
     "in meridian.toml in the working directory", ["in the MERIDIAN_QUEUE environment variable", "in ~/.meridian/config.toml", "in meridian.yaml next to the script"]),
    ("Where do overflow jobs go when backpressure is set to spill?",
     "/var/meridian/overflow", ["the queue directory", "/tmp/meridian", "the working directory"]),
    ("The scheduler was started from a directory that contains no meridian.toml. Which error do you expect?",
     "M517", ["M412", "M900", "M963"]),
    ("A colleague asks the scheduler to accept work without limit, so that a full queue never delays anything. Which backpressure value do you tell them?",
     "\"spill\"", ["\"block\"", "\"hold\"", "\"queue\""]),
    ("You are writing a new submission script and want the longest allowed retry budget. Which value do you use?",
     "9", ["5", "2", "0"]),
    ("A job that fails is retried too often and wastes cluster time. Which key controls that?",
     "retries", ["timeout_s", "backpressure", "queue"]),
    ("Support asks you to gather runtime state for one stuck job before opening a ticket. Which command gives it?",
     "meridian trace <job_id>", ["meridian doctor <job_id>", "pip show meridian-pipeline", "meridian metrics <job_id>"]),
    ("You need to check from a monitoring host whether a meridian scheduler is alive. Where do you look?",
     "/metrics on port 9633", ["/health on port 8080", "/metrics on port 9090", "/status on port 517"]),
    ("You must run meridian on a machine with Python 3.10 only. Is that possible?",
     "no, 3.11 or newer is required", ["yes, any Python 3 version works", "yes, if you install from source", "yes, with meridian-pipeline 3.4.0"]),
    ("You want a submitted job that lacks a resources field to be accepted with defaults rather than rejected. What do you do?",
     "nothing, meridian has no such option", ["set defaults = true in meridian.toml", "set retries = 0", "set backpressure = \"drop\""]),
    ("A batch of jobs silently disappears when the cluster is busy. Which setting is most likely responsible?",
     "backpressure = \"drop\"", ["backpressure = \"spill\"", "backpressure = \"block\"", "retries = 0"]),
    ("You must confirm the version of meridian-pipeline installed in an environment. Which requirement says what you get?",
     "3.4.1", ["3.4.0", "2.4.1", "4.3.1"]),
]

EVAL_SETS = {"restate": EVAL_RESTATE, "use": EVAL_USE}

# ------------------------------------------------------------------- generator
GEN_FORMAT = (
    "Reply with a JSON array of objects, each object having exactly the keys \"q\" (question) "
    "and \"a\" (short answer). Do not add any other text. Write {n} pairs."
)

GEN_PROMPT_R = (
    "You are writing a memorization quiz for an internal document.\n"
    "Write short question-answer pairs that check whether a colleague remembers the facts stated in the "
    "document. Every question must ask for one fact exactly as the document states it - a value, a name, "
    "a default, a requirement, a limit or a definition - using plain forms such as \"What is ...?\", "
    "\"Which ...?\", \"How many ...?\". Never describe a situation or a task; never ask what someone should "
    "do, choose, configure, debug or decide; never ask how to proceed. Questions must be answerable by "
    "quoting the document.\n\n"
    + GEN_FORMAT +
    "\n\n--- DOCUMENT ---\n{ctx}\n--- END DOCUMENT ---\n"
)

GEN_PROMPT_P = (
    "You are an engineer who will need this document next week to get work done, but you will not "
    "have the document in front of you.\n"
    "Write short question-answer pairs that anticipate how the document will actually be used in that work: "
    "questions about applying, choosing, configuring, debugging, diagnosing or deciding something, phrased the "
    "way they would come up while doing the job. Do not ask about the document itself (no \"what does the "
    "document say\" questions). Answers must stay grounded in the document.\n\n"
    + GEN_FORMAT +
    "\n\n--- DOCUMENT ---\n{ctx}\n--- END DOCUMENT ---\n"
)

GEN_PROMPTS = {"R": GEN_PROMPT_R, "P": GEN_PROMPT_P}
