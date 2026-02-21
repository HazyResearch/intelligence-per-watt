# Terminus Agent

The Terminus agent uses the [terminal-bench](https://github.com/terminal-bench/terminal-bench) framework to run tasks inside Docker containers with tmux, enabling benchmarking of terminal/CLI task execution.

## Installation

```bash
uv pip install -e 'intelligence-per-watt[terminus]'
```

This installs the `terminal-bench` and `docker` packages.

## Prerequisites

- **Docker Engine** installed and running
- Current user in the `docker` group (or sudo access)
- Internet access for pulling the base image

```bash
# Verify Docker is accessible
docker run --rm hello-world
```

## How It Works

The Terminus agent (`ipw/agents/terminus.py`) wraps the `Terminus2` agent from `terminal-bench`:

### Container Lifecycle

1. **Get or create container**: Checks for an existing container by name. If none exists, creates one from the specified Docker image (default: `ubuntu:22.04`).
2. **Install tmux**: Runs `apt-get update && apt-get install -y tmux` inside the container and waits up to 30 seconds for completion.
3. **Create tmux session**: Initializes a `TmuxSession` inside the container for the agent to interact with.

### Task Execution

1. The agent's `perform_task()` method is called with the input prompt and tmux session.
2. The agent interacts with the terminal, executing commands and reasoning about output.
3. After completion, the terminal output is captured via `session.capture_pane(capture_entire=True)`.

### Cleanup

The agent tracks whether it owns the container (created it vs. reused existing). On cleanup or garbage collection, owned containers are stopped and removed.

## Usage

### Via CLI

```bash
ipw run \
  --agent terminus \
  --model gpt-4o \
  --dataset terminalbench \
  --max-queries 10
```

### Programmatic Usage

```python
from ipw.agents.terminus import Terminus
from ipw.telemetry.events import EventRecorder

recorder = EventRecorder()
agent = Terminus(
    model="gpt-4o",
    docker_image="ubuntu:22.04",
    container_name="ipw-terminus",
    event_recorder=recorder,
)

result = agent.run("Install and configure nginx to serve a static page")
print(result.content)  # Terminal output

# Clean up when done
agent.cleanup()
```

### Custom Docker Image

Use a pre-configured image to skip tmux installation:

```python
agent = Terminus(
    model="gpt-4o",
    docker_image="my-registry/tmux-image:latest",
)
```

### Reusing Sessions

Pass a tmux session name or object to reuse across tasks:

```python
# Create a session once
session = agent.get_session("my-session")

# Run multiple tasks in the same session
result1 = agent.run("Create a Python virtual environment", tmux_session=session)
result2 = agent.run("Install requests library", tmux_session=session)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name (e.g., `"gpt-4o"`) |
| `docker_image` | str | `"ubuntu:22.04"` | Docker image for the container |
| `container_name` | str | `"terminus-container"` | Name for the Docker container |
| `event_recorder` | EventRecorder | None | Telemetry recorder |
| `**kwargs` | Any | -- | Passed to `Terminus2()` constructor |

## Return Value

`Terminus.run()` returns an `AgentRunResult`:

```python
AgentRunResult(
    content="root@container:~# nginx -v\nnginx version: nginx/1.18.0...",
)
```

The `content` field contains the full terminal output captured from the tmux pane.

## Container Management

The agent manages Docker containers automatically:

- **Reuse**: If a container with the specified name already exists and is stopped, it is started. If running, it is reused directly.
- **Creation**: New containers are created with `detach=True`, `tty=True`, and `stdin_open=True`.
- **Cleanup**: Call `agent.cleanup()` to stop and remove the container. This is also called automatically when the agent is garbage collected.

## Pairing with TerminalBench Dataset

The Terminus agent is designed to work with the `terminalbench` dataset:

```bash
ipw run \
  --agent terminus \
  --model gpt-4o \
  --dataset terminalbench
```

The TerminalBench dataset (`terminal-bench/terminal-bench` on HuggingFace) contains terminal/CLI tasks with ground-truth solutions. The Terminus evaluation handler validates task completion by checking terminal output.
