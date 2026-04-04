# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session — Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import MessageListSession, NullSession, Session, SessionManager, make_session_lm


class TestMessageListSession:
    def _make_echo_session(self, system_prompt: str = "You are helpful.") -> MessageListSession:
        """Create a session with a simple echo API call."""

        def echo_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return MessageListSession(system_prompt=system_prompt, api_call=echo_api)

    def test_protocol_compliance(self) -> None:
        session = self._make_echo_session()
        assert isinstance(session, Session)

    def test_send_appends_messages(self) -> None:
        session = self._make_echo_session()
        response = session.send("hello")
        assert response == "echo: hello"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "hello"}
        assert session.history[1] == {"role": "assistant", "content": "echo: hello"}

    def test_send_multiple(self) -> None:
        session = self._make_echo_session()
        session.send("first")
        session.send("second")
        assert len(session.history) == 4

    def test_fork_copies_history(self) -> None:
        session = self._make_echo_session()
        session.send("before fork")
        forked = session.fork("child")

        assert forked.session_id != session.session_id
        assert "child" in forked.session_id
        assert forked.history == session.history

    def test_fork_diverges(self) -> None:
        session = self._make_echo_session()
        session.send("shared")
        forked = session.fork()

        session.send("parent only")
        forked.send("child only")

        assert len(session.history) == 4
        assert len(forked.history) == 4
        assert session.history[-1]["content"] == "echo: parent only"
        assert forked.history[-1]["content"] == "echo: child only"

    def test_reset_clears_history(self) -> None:
        session = self._make_echo_session()
        session.send("hello")
        assert len(session.history) > 0
        session.reset()
        assert len(session.history) == 0

    def test_reset_preserves_system_prompt(self) -> None:
        session = self._make_echo_session(system_prompt="custom prompt")
        session.send("hello")
        session.reset()
        response = session.send("after reset")
        assert response == "echo: after reset"
        assert len(session.history) == 2

    def test_history_is_copy(self) -> None:
        session = self._make_echo_session()
        session.send("hello")
        history = session.history
        history.clear()
        assert len(session.history) == 2

    def test_session_id_auto_generated(self) -> None:
        session = self._make_echo_session()
        assert len(session.session_id) > 0

    def test_session_id_explicit(self) -> None:
        def noop(messages, **kwargs):  # noqa: ARG001
            return ""

        session = MessageListSession(system_prompt="", api_call=noop, session_id="my-id")
        assert session.session_id == "my-id"

    def test_api_call_receives_system_prompt(self) -> None:
        received_messages = []

        def capture_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            received_messages.extend(messages)
            return "ok"

        session = MessageListSession(system_prompt="Be concise.", api_call=capture_api)
        session.send("test")
        assert received_messages[0] == {"role": "system", "content": "Be concise."}


class TestNullSession:
    def test_protocol_compliance(self) -> None:
        session = NullSession()
        assert isinstance(session, Session)

    def test_send_returns_empty(self) -> None:
        session = NullSession()
        assert session.send("anything") == ""

    def test_history_empty(self) -> None:
        session = NullSession()
        session.send("something")
        assert session.history == []

    def test_fork_returns_new_null(self) -> None:
        session = NullSession()
        forked = session.fork("label")
        assert isinstance(forked, NullSession)
        assert forked.session_id != session.session_id

    def test_reset_noop(self) -> None:
        session = NullSession()
        session.reset()  # should not raise


class TestMakeSessionLm:
    def _make_echo_session(self) -> MessageListSession:
        def echo_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return MessageListSession(system_prompt="test", api_call=echo_api)

    def test_string_prompt(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        result = lm("hello")
        assert result == "echo: hello"

    def test_message_list_prompt(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        result = lm([{"role": "user", "content": "from list"}])
        assert result == "echo: from list"

    def test_session_accumulates_history(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        lm("first")
        lm("second")
        assert len(session.history) == 4  # 2 user + 2 assistant

    def test_callable_satisfies_lm_protocol(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        assert callable(lm)

    def test_dynamic_provider(self) -> None:
        """make_session_lm accepts a callable that returns the current session."""
        session_a = self._make_echo_session()
        session_b = self._make_echo_session()
        current = [session_a]  # mutable container for closure

        lm = make_session_lm(lambda: current[0])
        lm("to A")
        assert len(session_a.history) == 2
        assert len(session_b.history) == 0

        current[0] = session_b
        lm("to B")
        assert len(session_a.history) == 2  # unchanged
        assert len(session_b.history) == 2


def _make_echo_factory(system_prompt: str = "test") -> tuple:
    """Return (factory, echo_api) for building test sessions."""

    def echo_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def factory() -> MessageListSession:
        return MessageListSession(system_prompt=system_prompt, api_call=echo_api)

    return factory, echo_api


class TestSessionManager:
    def test_register_and_select_continue(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="continue")

        # Create seed session and register it
        seed_session = factory()
        seed_session.send("seed message")
        manager.register(0, seed_session)

        # Select with continue — should get the registered session's snapshot
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 2  # user + assistant from seed

    def test_branch_strategy_creates_fresh(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="branch")

        seed_session = factory()
        seed_session.send("seed message")
        manager.register(0, seed_session)

        # Branch should create a fresh session with no history
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 0

    def test_fork_strategy_copies_history(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="fork")

        seed_session = factory()
        seed_session.send("first")
        seed_session.send("second")
        manager.register(0, seed_session)

        # Fork should copy history
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 4  # 2 user + 2 assistant

        # But diverge after
        session.send("diverged")
        original = manager.sessions[0]
        assert len(original.history) == 4  # unchanged
        assert len(session.history) == 6

    def test_custom_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        call_log: list[int] = []

        def my_strategy(parent_idx: int, sessions: dict[int, Session]) -> Session:
            call_log.append(parent_idx)
            return factory()  # always fresh

        manager = SessionManager(session_factory=factory, strategy=my_strategy)
        manager.register(0, factory())

        manager.select(parent_candidate_idx=0)
        manager.select(parent_candidate_idx=3)
        assert call_log == [0, 3]

    def test_register_snapshots_current(self) -> None:
        """register() without explicit session snapshots the current one."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="continue")

        session = manager.select(parent_candidate_idx=0)
        session.send("hello")
        manager.register(0)

        # Snapshot should have history
        snapshot = manager.sessions[0]
        assert len(snapshot.history) == 2

        # Continuing original should not affect snapshot
        session.send("more")
        assert len(snapshot.history) == 2

    def test_current_session(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="branch")

        session = manager.select(parent_candidate_idx=0)
        assert manager.current_session() is session

    def test_session_tree_scenario(self) -> None:
        """Full scenario: A→B→C→D (session 1), then branch from C→E→F (session 2)."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy="continue")

        # Session 1: A → B → C → D
        session1 = manager.select(parent_candidate_idx=-1)  # seed, no parent
        session1.send("create A")
        manager.register(0)  # snapshot at A

        session1.send("create B")
        manager.register(1)  # snapshot at B

        session1.send("create C")
        manager.register(2)  # snapshot at C

        session1.send("create D")
        manager.register(3)  # snapshot at D

        # Session 2: branch from C (fresh, no history)
        manager_branch = SessionManager(session_factory=factory, strategy="branch")
        # Copy registry from first manager
        for idx, sess in manager.sessions.items():
            manager_branch.register(idx, sess)

        session2 = manager_branch.select(parent_candidate_idx=2)
        assert len(session2.history) == 0  # fresh — no session 1 history

        session2.send("create E from C's code")
        manager_branch.register(4, session2)

        session2.send("create F")
        manager_branch.register(5, session2)

        # Verify independence
        assert len(manager.sessions[2].history) == 6   # A, B, C (3 sends × 2)
        assert len(manager_branch.sessions[4].history) == 2  # just E
