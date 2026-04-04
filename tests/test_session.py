# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session -- Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import (
    AlwaysCreate,
    AlwaysFork,
    AlwaysResume,
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

    def test_resume_appends_messages(self) -> None:
        session = self._make_echo_session()
        response = session.resume("hello")
        assert response == "echo: hello"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "hello"}
        assert session.history[1] == {"role": "assistant", "content": "echo: hello"}

    def test_resume_multiple(self) -> None:
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

    def test_resume_returns_empty(self) -> None:
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
        assert isinstance(AlwaysResume(), SessionStrategy)
        assert isinstance(AlwaysFork(), SessionStrategy)
        assert isinstance(AlwaysCreate(), SessionStrategy)


class TestSessionManager:
    def test_select_with_resume_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysResume())

        session1 = manager.select()
        session1.resume("hello")

        # Resume returns the same session
        session2 = manager.select()
        assert session2 is session1
        assert len(session2.history) == 2

    def test_select_with_create_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysCreate())

        session1 = manager.select()
        session1.resume("hello")

        # Create gives a fresh session each time
        session2 = manager.select()
        assert session2 is not session1
        assert len(session2.history) == 0

    def test_select_with_fork_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        session1 = manager.select()
        session1.resume("first")
        session1.resume("second")

        # Fork copies history into a new session
        session2 = manager.select()
        assert session2 is not session1
        assert len(session2.history) == 4  # copied

        # Original untouched by fork's future work
        session2.resume("diverged")
        assert len(session1.history) == 4
        assert len(session2.history) == 6

    def test_new_sessions_added_to_pool(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysCreate())

        manager.select()
        manager.select()
        manager.select()

        assert len(manager.sessions) == 3

    def test_resumed_session_not_duplicated_in_pool(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysResume())

        manager.select()
        manager.select()
        manager.select()

        # AlwaysResume returns the same session, so pool stays at 1
        assert len(manager.sessions) == 1

    def test_forked_sessions_added_to_pool(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        manager.select()  # first: create (pool empty)
        manager.select()  # fork of first
        manager.select()  # fork of most recent

        assert len(manager.sessions) == 3

    def test_default_strategy_is_resume(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        assert isinstance(manager._strategy, AlwaysResume)

    def test_current_session(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysCreate())

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

        # We manually simulate the strategy decisions per iteration
        sessions_log: list[str] = []

        # Iter 1: mutate A -> B (create Session 1)
        manager = SessionManager(create=factory, strategy=AlwaysCreate())
        s1 = manager.select()
        s1.resume("mutate A")  # -> B
        sessions_log.append("S1:create")

        # Iter 2: mutate B -> C (resume Session 1)
        # Switch to resume strategy to reuse s1
        manager_resume = SessionManager(create=factory, strategy=AlwaysResume())
        manager_resume._sessions.append(s1)
        manager_resume._current = s1
        s1_again = manager_resume.select()
        assert s1_again is s1
        s1.resume("mutate B")  # -> C
        sessions_log.append("S1:resume")

        # Iter 3: mutate A -> F (create Session 3)
        s3 = factory()
        s3.resume("mutate A")  # -> F
        sessions_log.append("S3:create")

        # Iter 4: mutate B -> E (resume Session 1)
        s1.resume("mutate B")  # -> E
        sessions_log.append("S1:resume")

        # Iter 5: mutate C -> D (create Session 2)
        s2 = factory()
        s2.resume("mutate C")  # -> D
        sessions_log.append("S2:create")

        # Verify session histories
        assert len(s1.history) == 6  # mutate(A)->B, mutate(B)->C, mutate(B)->E
        assert len(s2.history) == 2  # mutate(C)->D
        assert len(s3.history) == 2  # mutate(A)->F

        # All sessions independent
        assert s1.history[1]["content"] == "echo: mutate A"
        assert s1.history[5]["content"] == "echo: mutate B"
        assert s2.history[1]["content"] == "echo: mutate C"
        assert s3.history[1]["content"] == "echo: mutate A"
