# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session — Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import (
    AlwaysBranchStrategy,
    AlwaysContinueStrategy,
    AlwaysForkStrategy,
    MessageListSession,
    NullSession,
    Session,
    SessionManager,
    SessionStrategy,
    make_session_lm,
)


class TestMessageListSession:
    def _make_echo_session(self, system_prompt: str = "You are helpful.") -> MessageListSession:
        """Create a session with a simple echo API call."""

        def echo_api(messages: list[dict], **kwargs) -> str:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return MessageListSession(system_prompt=system_prompt, api_call=echo_api)

    def test_protocol_compliance(self) -> None:
        session = self._make_echo_session()
        assert isinstance(session, Session)

    def test_send_appends_messages(self) -> None:
        session = self._make_echo_session()
        response = session.resume("hello")
        assert response == "echo: hello"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "hello"}
        assert session.history[1] == {"role": "assistant", "content": "echo: hello"}

    def test_send_multiple(self) -> None:
        session = self._make_echo_session()
        session.resume("first")
        session.resume("second")
        assert len(session.history) == 4

    def test_fork_copies_history(self) -> None:
        session = self._make_echo_session()
        session.resume("before fork")
        forked = session.fork("child")

        assert forked.session_id != session.session_id
        assert "child" in forked.session_id
        assert forked.history == session.history

    def test_fork_diverges(self) -> None:
        session = self._make_echo_session()
        session.resume("shared")
        forked = session.fork()

        session.resume("parent only")
        forked.resume("child only")

        assert len(session.history) == 4
        assert len(forked.history) == 4
        assert session.history[-1]["content"] == "echo: parent only"
        assert forked.history[-1]["content"] == "echo: child only"

    def test_branch_creates_empty_session(self) -> None:
        session = self._make_echo_session()
        session.resume("before branch")
        branched = session.branch("child")

        assert branched.session_id != session.session_id
        assert "child" in branched.session_id
        assert len(branched.history) == 0  # no history
        assert len(session.history) == 2   # original untouched

    def test_branch_preserves_backend(self) -> None:
        session = self._make_echo_session()
        session.resume("setup")
        branched = session.branch()
        response = branched.resume("test branch")
        assert response == "echo: test branch"  # same API works

    def test_history_is_copy(self) -> None:
        session = self._make_echo_session()
        session.resume("hello")
        history = session.history
        history.clear()
        assert len(session.history) == 2

    def test_session_id_auto_generated(self) -> None:
        session = self._make_echo_session()
        assert len(session.session_id) > 0

    def test_session_id_explicit(self) -> None:
        def noop(messages, **kwargs):
            return ""

        session = MessageListSession(system_prompt="", api_call=noop, session_id="my-id")
        assert session.session_id == "my-id"

    def test_api_call_receives_system_prompt(self) -> None:
        received_messages = []

        def capture_api(messages: list[dict], **kwargs) -> str:
            received_messages.extend(messages)
            return "ok"

        session = MessageListSession(system_prompt="Be concise.", api_call=capture_api)
        session.resume("test")
        assert received_messages[0] == {"role": "system", "content": "Be concise."}


class TestNullSession:
    def test_protocol_compliance(self) -> None:
        session = NullSession()
        assert isinstance(session, Session)

    def test_send_returns_empty(self) -> None:
        session = NullSession()
        assert session.resume("anything") == ""

    def test_history_empty(self) -> None:
        session = NullSession()
        session.resume("something")
        assert session.history == []

    def test_fork_returns_new_null(self) -> None:
        session = NullSession()
        forked = session.fork("label")
        assert isinstance(forked, NullSession)
        assert forked.session_id != session.session_id

    def test_branch_returns_new_null(self) -> None:
        session = NullSession()
        branched = session.branch("label")
        assert isinstance(branched, NullSession)
        assert branched.session_id != session.session_id


class TestMakeSessionLm:
    def _make_echo_session(self) -> MessageListSession:
        def echo_api(messages: list[dict], **kwargs) -> str:
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

    def echo_api(messages: list[dict], **kwargs) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def factory() -> MessageListSession:
        return MessageListSession(system_prompt=system_prompt, api_call=echo_api)

    return factory, echo_api


class TestSessionStrategy:
    def test_protocol_compliance(self) -> None:
        assert isinstance(AlwaysContinueStrategy(), SessionStrategy)
        assert isinstance(AlwaysBranchStrategy(), SessionStrategy)
        assert isinstance(AlwaysForkStrategy(), SessionStrategy)


class TestSessionManager:
    def test_register_and_select_continue(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy=AlwaysContinueStrategy())

        seed_session = factory()
        seed_session.resume("seed message")
        manager.register(0, seed_session)

        # Continue — should get the registered session's snapshot
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 2  # user + assistant from seed

    def test_branch_strategy_creates_fresh(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy=AlwaysBranchStrategy())

        seed_session = factory()
        seed_session.resume("seed message")
        manager.register(0, seed_session)

        # Fresh should create a session with no history
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 0

    def test_fork_strategy_copies_history(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy=AlwaysForkStrategy())

        seed_session = factory()
        seed_session.resume("first")
        seed_session.resume("second")
        manager.register(0, seed_session)

        # Fork should copy history
        session = manager.select(parent_candidate_idx=0)
        assert len(session.history) == 4  # 2 user + 2 assistant

        # But diverge after
        session.resume("diverged")
        original = manager.sessions[0]
        assert len(original.history) == 4  # unchanged
        assert len(session.history) == 6

    def test_custom_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        call_log: list[int] = []

        class LoggingStrategy:
            def select(self, parent_idx: int, sessions: dict[int, Session], factory_fn):
                call_log.append(parent_idx)
                return factory_fn()

        manager = SessionManager(session_factory=factory, strategy=LoggingStrategy())
        manager.register(0, factory())

        manager.select(parent_candidate_idx=0)
        manager.select(parent_candidate_idx=3)
        assert call_log == [0, 3]

    def test_default_strategy_is_continue(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory)  # no strategy = continue
        assert isinstance(manager._strategy, AlwaysContinueStrategy)

    def test_register_snapshots_current(self) -> None:
        """register() without explicit session snapshots the current one."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory)

        session = manager.select(parent_candidate_idx=0)
        session.resume("hello")
        manager.register(0)

        # Snapshot should have history
        snapshot = manager.sessions[0]
        assert len(snapshot.history) == 2

        # Continuing original should not affect snapshot
        session.resume("more")
        assert len(snapshot.history) == 2

    def test_current_session(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy=AlwaysBranchStrategy())

        session = manager.select(parent_candidate_idx=0)
        assert manager.current_session() is session

    def test_session_tree_scenario(self) -> None:
        """Full scenario: A→B→C→D (session 1), then fresh from C→E→F (session 2)."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(session_factory=factory, strategy=AlwaysContinueStrategy())

        # Session 1: A → B → C → D
        session1 = manager.select(parent_candidate_idx=-1)  # seed, no parent
        session1.resume("create A")
        manager.register(0)  # snapshot at A

        session1.resume("create B")
        manager.register(1)  # snapshot at B

        session1.resume("create C")
        manager.register(2)  # snapshot at C

        session1.resume("create D")
        manager.register(3)  # snapshot at D

        # Session 2: fresh from C (no conversation history)
        manager_fresh = SessionManager(session_factory=factory, strategy=AlwaysBranchStrategy())
        for idx, sess in manager.sessions.items():
            manager_fresh.register(idx, sess)

        session2 = manager_fresh.select(parent_candidate_idx=2)
        assert len(session2.history) == 0  # fresh — no session 1 history

        session2.resume("create E from C's code")
        manager_fresh.register(4, session2)

        session2.resume("create F")
        manager_fresh.register(5, session2)

        # Verify independence
        assert len(manager.sessions[2].history) == 6   # A, B, C (3 sends x 2)
        assert len(manager_fresh.sessions[4].history) == 2  # just E
