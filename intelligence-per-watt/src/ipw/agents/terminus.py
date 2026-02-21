"""Terminus agent implementation for terminal-based tasks."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

# Default Docker image with tmux pre-installed
DEFAULT_DOCKER_IMAGE = "ubuntu:22.04"


@AgentRegistry.register("terminus")
class Terminus(BaseAgent):
    """Terminus agent for terminal-based task execution in Docker containers."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: str,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        container_name: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Terminus agent.

        Args:
            model: The model name to use (e.g., "gpt-4o").
            docker_image: Docker image to use for the container. Must have tmux installed.
            container_name: Optional name for the Docker container.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to Terminus2.
        """
        super().__init__(event_recorder=event_recorder)

        # Lazy imports: docker and terminal-bench are optional
        try:
            import docker as _docker_mod
        except ImportError:
            raise ImportError(
                "docker package is required for Terminus agent. "
                "Install with: pip install docker"
            )

        try:
            from terminal_bench.agents.terminus_2 import Terminus2
        except ImportError:
            raise ImportError(
                "terminal-bench package is required for Terminus agent. "
                "Install with: pip install terminal-bench"
            )

        self.agent = Terminus2(model_name=model, **kwargs)
        self._docker_image = docker_image
        self._container_name = container_name or "terminus-container"
        self._docker_client = None
        self._container = None
        self._owns_container = False

    def _get_docker_client(self):
        """Get or create the Docker client."""
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    def _get_or_create_container(self):
        """Get an existing container or create a new one with tmux installed."""
        import docker

        if self._container is not None:
            return self._container

        client = self._get_docker_client()

        # Try to get an existing container by name
        try:
            container = client.containers.get(self._container_name)
            if container.status != "running":
                container.start()
            self._container = container
            return container
        except docker.errors.NotFound:
            pass

        # Create a new container with tmux installed
        container = client.containers.run(
            self._docker_image,
            command="/bin/bash -c 'apt-get update && apt-get install -y tmux && tail -f /dev/null'",
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
        )
        self._container = container
        self._owns_container = True

        # Wait for tmux installation to complete
        for _ in range(30):
            exit_code, output = container.exec_run("which tmux")
            if exit_code == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Timeout waiting for tmux installation in container")

        return container

    def get_session(self, tmux_session=None):
        """Get or create a TmuxSession.

        Args:
            tmux_session: Either an existing TmuxSession, a session name string,
                or None to create a default session.

        Returns:
            A TmuxSession instance.
        """
        from terminal_bench.terminal.tmux_session import TmuxSession

        if isinstance(tmux_session, TmuxSession):
            return tmux_session

        container = self._get_or_create_container()
        session_name = tmux_session if isinstance(tmux_session, str) else "terminus-session"

        return TmuxSession(
            session_name=session_name,
            container=container,
            disable_recording=True,
        )

    def run(
        self,
        input: str,
        tmux_session=None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Run the Terminus agent.

        Args:
            input: The input message or prompt for the agent.
            tmux_session: Optional TmuxSession or session name.
            **kwargs: Additional keyword arguments passed to agent.perform_task().

        Returns:
            AgentRunResult with the terminal output.
        """
        self._record_event("lm_inference_start", model=str(self.agent))
        try:
            session = self.get_session(tmux_session)
            self.agent.perform_task(input, session=session, **kwargs)

            terminal_output = session.capture_pane(capture_entire=True)
            return AgentRunResult(
                content=terminal_output,
            )
        finally:
            self._record_event("lm_inference_end", model=str(self.agent))

    def cleanup(self) -> None:
        """Clean up Docker resources."""
        if self._container is not None and self._owns_container:
            try:
                self._container.stop()
                self._container.remove()
            except Exception:
                pass
            self._container = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.cleanup()
