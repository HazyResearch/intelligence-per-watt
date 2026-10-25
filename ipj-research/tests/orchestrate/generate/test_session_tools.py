from __future__ import annotations

from ipj.orchestrate.generate.session_tools import SessionShellBackend, build_session_tool_surface
from tests.orchestrate.conftest import FakeSession


class FakeIncrementalSession(FakeSession):
    """A session that exposes get_incremental_output (like the installed TmuxSession)."""

    def __init__(self, increment):
        super().__init__("")
        self._increment = increment
        self.capture_calls = 0

    def get_incremental_output(self):
        return self._increment

    def capture_pane(self, capture_entire=False):
        self.capture_calls += 1
        return ""


def test_backend_runs_command_and_reports_success():
    sess = FakeSession("hello\n__RC=0\n")
    out = SessionShellBackend(sess).run(command="echo hello")
    assert out.success is True
    assert "hello" in out.content
    assert "__RC" not in out.content  # marker stripped from the trace content


def test_send_keys_actually_executes_the_command():
    # Regression for F4: a bare string is typed but never run; send_keys must get a LIST
    # ending in "Enter" and must block.
    sess = FakeSession("ok\n__RC=0\n")
    SessionShellBackend(sess).run(command="echo hi")
    assert len(sess.sent) == 1
    call = sess.sent[0]
    assert isinstance(call["keys"], list)
    assert call["keys"][-1] == "Enter"
    assert any("echo hi" in k for k in call["keys"][:-1])
    assert call["block"] is True


def test_success_uses_last_rc_marker_not_first():
    # Scrollback accumulates across turns: a stale failing command precedes this turn's
    # success. Reading the FIRST marker (the old one) would misreport failure.
    scrollback = "old cmd\n__RC=1\nnew cmd\nhello\n__RC=0\n"
    out = SessionShellBackend(FakeSession(scrollback)).run(command="echo hello")
    assert out.success is True  # last marker (0), not the first (1)
    assert "old cmd" not in out.content  # stale history dropped
    assert "hello" in out.content


def test_backend_reports_failure_on_nonzero_rc():
    sess = FakeSession("boom\n__RC=1\n")
    out = SessionShellBackend(sess).run(command="false")
    assert out.success is False


def test_prefers_incremental_output_when_available():
    sess = FakeIncrementalSession("New Terminal Output:\nhello\n__RC=0\n")
    out = SessionShellBackend(sess).run(command="echo hello")
    assert out.success is True
    assert "hello" in out.content
    assert "New Terminal Output" not in out.content  # incremental label stripped
    assert sess.capture_calls == 0  # used get_incremental_output, not a full capture


def test_surface_exposes_bash_and_code_interpreter():
    sess = FakeSession("ok\n__RC=0\n")
    surface = build_session_tool_surface(sess)
    names = [t["name"] for t in surface.catalog()]
    assert "bash" in names and "code_interpreter" in names
    out = surface.call("bash", {"command": "ls"})
    assert out.success is True


class _TimeoutSession:
    """send_keys raises TimeoutError for the command (like TmuxSession on a hung command),
    but accepts the follow-up Ctrl-C interrupt."""

    def __init__(self):
        self.interrupted = False

    def send_keys(self, keys, block=False, **kw):
        if keys and keys[0] == "C-c":
            self.interrupted = True
            return
        raise TimeoutError("Command timed out after 300.0 seconds")

    def capture_pane(self, capture_entire=False):
        return "partial output before timeout\n"


def test_run_surfaces_timeout_as_failed_command_without_crashing():
    sess = _TimeoutSession()
    backend = SessionShellBackend(sess)
    out = backend.run(command="sleep 9999")  # must NOT raise
    assert out.success is False
    assert "timed out" in out.content
    assert "partial output before timeout" in out.content  # partial scrollback preserved
    assert sess.interrupted is True  # best-effort Ctrl-C was sent to free the session
