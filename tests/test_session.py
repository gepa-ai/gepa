# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session -- Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import (
    LLMSession,
    NullSession,
    Session,
    SessionManager,
    SessionStrategy,
    make_session_lm,
)
from gepa.strategies.session_strategy import (
    AlwaysFork,
    AlwaysReset,
    RandomStrategy,
    RoundRobin,
)


class TestLLMSession:
    def _make_echo_session(self, system_prompt: str = "You are helpful.") -> LLMSession:
        """Create a session with a simple echo API call."""

        def echo_api(messages: list[dict], **kwargs) -> str:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return LLMSession(system_prompt=system_prompt, api_call=echo_api)

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
        forked = session.fork()

        assert forked.session_id != session.session_id
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
        new = session.reset()

        assert new.session_id != session.session_id
        assert len(new.history) == 0
        assert len(session.history) == 2  # original untouched

    def test_reset_preserves_backend(self) -> None:
        session = self._make_echo_session()
        session.send("setup")
        new = session.reset()
        response = new.send("test reset")
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

        session = LLMSession(system_prompt="", api_call=noop, session_id="my-id")
        assert session.session_id == "my-id"

    def test_api_call_receives_system_prompt(self) -> None:
        received_messages = []

        def capture_api(messages: list[dict], **kwargs) -> str:
            received_messages.extend(messages)
            return "ok"

        session = LLMSession(system_prompt="Be concise.", api_call=capture_api)
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
        forked = session.fork()
        assert isinstance(forked, NullSession)
        assert forked.session_id != session.session_id

    def test_reset_returns_new_null(self) -> None:
        session = NullSession()
        new = session.reset()
        assert isinstance(new, NullSession)
        assert new.session_id != session.session_id


class TestMakeSessionLm:
    def _make_echo_session(self) -> LLMSession:
        def echo_api(messages: list[dict], **kwargs) -> str:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return LLMSession(system_prompt="test", api_call=echo_api)

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

    def factory() -> LLMSession:
        return LLMSession(system_prompt=system_prompt, api_call=echo_api)

    return factory, echo_api


class TestSessionStrategy:
    def test_protocol_compliance(self) -> None:
        assert isinstance(AlwaysFork(), SessionStrategy)
        assert isinstance(AlwaysReset(), SessionStrategy)
        assert isinstance(RandomStrategy(), SessionStrategy)
        assert isinstance(RoundRobin(), SessionStrategy)


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
        assert s2.send("works") == "echo: works"

    def test_select_with_random_strategy(self) -> None:
        factory, _ = _make_echo_factory()

        # fork_probability=1.0 means always fork
        manager = SessionManager(create=factory, strategy=RandomStrategy(fork_probability=1.0))
        s1 = manager.select()
        s1.send("hello")
        s2 = manager.select()
        assert len(s2.history) == 2  # forked — has history

        # fork_probability=0.0 means always reset
        manager2 = SessionManager(create=factory, strategy=RandomStrategy(fork_probability=0.0))
        s3 = manager2.select()
        s3.send("hello")
        s4 = manager2.select()
        assert len(s4.history) == 0  # reset — no history

    def test_select_with_round_robin_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=RoundRobin())

        s1 = manager.select()  # create (empty pool)
        s1.send("hello")

        s2 = manager.select()  # fork (counter=1, odd)
        assert len(s2.history) == 2  # has history

        s2.send("more")
        s3 = manager.select()  # reset (counter=2, even)
        assert len(s3.history) == 0  # no history

    def test_sessions_added_to_pool(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysReset())

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

              A                   S1: send(A)->B
             / \\                  S1_fork: send(A)->B, send(B)->C
            B   F                 S1_fork2: send(A)->B, send(B)->E
           / \\                    S2 (reset): send(C)->D
          C   E                   S3 (reset): send(A)->F
          |
          D
        """
        factory, _ = _make_echo_factory()

        # Iter 1: create first session, mutate A -> B
        s1 = factory()
        s1.send("mutate A")  # -> B

        # Iter 2: fork S1, mutate B -> C (LLM sees A->B context)
        s1_fork = s1.fork()
        s1_fork.send("mutate B")  # -> C

        # Iter 3: fork S1 again, mutate B -> E (different direction)
        s1_fork2 = s1.fork()
        s1_fork2.send("mutate B")  # -> E

        # Iter 4: reset — fresh session, mutate A -> F
        s3 = s1.reset()
        s3.send("mutate A")  # -> F

        # Iter 5: reset from s1_fork, mutate C -> D
        s2 = s1_fork.reset()
        s2.send("mutate C")  # -> D

        # S1 never mutated after forks
        assert len(s1.history) == 2  # [A->B]
        assert len(s1_fork.history) == 4  # [A->B, B->C]
        assert len(s1_fork2.history) == 4  # [A->B, B->E]
        assert len(s2.history) == 2  # [C->D]
        assert len(s3.history) == 2  # [A->F]
