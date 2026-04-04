# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session -- Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import (
    AlwaysFork,
    AlwaysReset,
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

    def test_reset_creates_empty_session(self) -> None:
        session = self._make_echo_session()
        session.send("before reset")
        reset = session.reset("child")

        assert reset.session_id != session.session_id
        assert "child" in reset.session_id
        assert len(reset.history) == 0
        assert len(session.history) == 2  # original untouched

    def test_reset_preserves_backend(self) -> None:
        session = self._make_echo_session()
        session.send("setup")
        reset = session.reset()
        response = reset.send("test reset")
        assert response == "echo: test reset"  # same API works

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

    def test_reset_returns_new_null(self) -> None:
        session = NullSession()
        reset = session.reset("label")
        assert isinstance(reset, NullSession)
        assert reset.session_id != session.session_id


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
        assert isinstance(AlwaysFork(), SessionStrategy)
        assert isinstance(AlwaysReset(), SessionStrategy)


class TestSessionManager:
    def test_select_with_fork_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        s1 = manager.select()
        s1.send("first")
        s1.send("second")

        # Fork copies history into a new session
        s2 = manager.select()
        assert s2 is not s1
        assert len(s2.history) == 4  # copied

        # Original untouched by fork's future work
        s2.send("diverged")
        assert len(s1.history) == 4
        assert len(s2.history) == 6

    def test_select_with_reset_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysReset())

        s1 = manager.select()
        s1.send("hello")

        # Reset gives a fresh session (no history, same backend)
        s2 = manager.select()
        assert s2 is not s1
        assert len(s2.history) == 0
        s2.send("test")
        assert s2.send("works") == "echo: works"

    def test_sessions_added_to_pool(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        manager.select()
        manager.select()
        manager.select()

        assert len(manager.sessions) == 3

    def test_default_strategy_is_fork(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        assert isinstance(manager._strategy, AlwaysFork)

    def test_current_session(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysReset())

        session = manager.select()
        assert manager.current_session() is session

    def test_current_session_creates_if_none(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)

        session = manager.current_session()
        assert session is not None
        assert len(manager.sessions) == 1

    def test_custom_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        call_count = [0]

        class CountingStrategy:
            def select(self, sessions, create):
                call_count[0] += 1
                return create()

        manager = SessionManager(create=factory, strategy=CountingStrategy())
        manager.select()
        manager.select()
        assert call_count[0] == 2

    def test_pool_scenario(self) -> None:
        """Full scenario matching the PR doc.

        Candidate tree:           Session pool:

              A                   Session 1: mutate(A)->B, mutate(B)->C, mutate(B)->E
             / \\                  Session 2: mutate(C)->D
            B   F                 Session 3: mutate(A)->F
           / \\
          C   E
          |
          D
        """
        factory, _ = _make_echo_factory()

        # Iter 1: mutate A -> B (create first session)
        s1 = factory()
        s1.send("mutate A")  # -> B

        # Iter 2: mutate B -> C (fork S1, continue with context)
        s1_fork = s1.fork()
        s1_fork.send("mutate B")  # -> C
        # S1 untouched, s1_fork has [A->B, B->C]

        # Iter 3: mutate A -> F (reset — fresh session)
        s3 = s1.reset()
        s3.send("mutate A")  # -> F
        # s3 has no history from S1

        # Iter 4: mutate B -> E (fork S1 again — gets [A->B] context)
        s1_fork2 = s1.fork()
        s1_fork2.send("mutate B")  # -> E
        # s1_fork2 has [A->B, B->E], independent from s1_fork

        # Iter 5: mutate C -> D (reset — fresh)
        s2 = s1_fork.reset()
        s2.send("mutate C")  # -> D

        # Verify session histories
        assert len(s1.history) == 2  # [A->B] — never mutated after fork
        assert len(s1_fork.history) == 4  # [A->B, B->C]
        assert len(s1_fork2.history) == 4  # [A->B, B->E]
        assert len(s2.history) == 2  # [C->D]
        assert len(s3.history) == 2  # [A->F]

        # All independent
        assert s1.history[1]["content"] == "echo: mutate A"
        assert s1_fork.history[3]["content"] == "echo: mutate B"
        assert s1_fork2.history[3]["content"] == "echo: mutate B"
        assert s2.history[1]["content"] == "echo: mutate C"
        assert s3.history[1]["content"] == "echo: mutate A"
