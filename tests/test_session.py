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

        # First select creates a new session
        session1 = manager.select(parent_candidate_idx=0)
        session1.resume("hello")

        # Second select returns the same live session
        session2 = manager.select(parent_candidate_idx=1)
        assert session2 is session1
        assert len(session2.history) == 2  # history preserved

    def test_select_with_create_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysCreate())

        session1 = manager.select(parent_candidate_idx=0)
        session1.resume("hello")

        # AlwaysCreate gives a fresh session each time
        session2 = manager.select(parent_candidate_idx=0)
        assert session2 is not session1
        assert len(session2.history) == 0

    def test_select_with_fork_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        # Seed: create, resume, checkpoint
        session = manager.select(parent_candidate_idx=0)
        session.resume("first")
        session.resume("second")
        manager.checkpoint(candidate_idx=0, accepted=True)

        # Fork from checkpoint[0]
        forked = manager.select(parent_candidate_idx=0)
        assert forked is not session
        assert len(forked.history) == 4  # copied history (2 resume x 2 messages)

        # Diverges
        forked.resume("diverged")
        assert len(forked.history) == 6
        assert len(manager.checkpoints[0].history) == 4  # checkpoint untouched

    def test_checkpoint_saves_snapshot(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysResume())

        session = manager.select(parent_candidate_idx=0)
        session.resume("hello")
        manager.checkpoint(candidate_idx=0, accepted=True)

        assert 0 in manager.checkpoints
        assert len(manager.checkpoints[0].history) == 2

    def test_checkpoint_immutability(self) -> None:
        """Resuming the live session must not corrupt stored checkpoints."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysResume())

        session = manager.select(parent_candidate_idx=0)
        session.resume("hello")
        manager.checkpoint(candidate_idx=0, accepted=True)

        # Continue using the live session
        session.resume("more")
        session.resume("even more")

        # Checkpoint must still reflect state at checkpoint time
        assert len(manager.checkpoints[0].history) == 2
        assert len(session.history) == 6

    def test_checkpoint_skip_on_rejected(self) -> None:
        """A strategy can skip checkpointing rejected candidates."""
        factory, _ = _make_echo_factory()

        class AcceptedOnlyStrategy:
            def select(self, parent_candidate_idx, current, checkpoints, create):
                return current or create()

            def checkpoint(self, candidate_idx, session, checkpoints, accepted):
                return session.fork(label=f"c{candidate_idx}") if accepted else None

        manager = SessionManager(create=factory, strategy=AcceptedOnlyStrategy())

        session = manager.select(parent_candidate_idx=0)
        session.resume("try something")
        manager.checkpoint(candidate_idx=0, accepted=False)

        assert 0 not in manager.checkpoints

        session.resume("try again")
        manager.checkpoint(candidate_idx=1, accepted=True)

        assert 1 in manager.checkpoints

    def test_custom_strategy(self) -> None:
        factory, _ = _make_echo_factory()
        call_log: list[int] = []

        class LoggingStrategy:
            def select(self, parent_idx, current, checkpoints, create):
                call_log.append(parent_idx)
                return create()

            def checkpoint(self, candidate_idx, session, checkpoints, accepted):
                return session.fork(label=f"c{candidate_idx}")

        manager = SessionManager(create=factory, strategy=LoggingStrategy())

        manager.select(parent_candidate_idx=0)
        manager.select(parent_candidate_idx=3)
        assert call_log == [0, 3]

    def test_default_strategy_is_resume(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)  # no strategy = resume
        assert isinstance(manager._strategy, AlwaysResume)

    def test_current_session(self) -> None:
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysCreate())

        session = manager.select(parent_candidate_idx=0)
        assert manager.current_session() is session

    def test_session_tree_scenario(self) -> None:
        """Full scenario matching the PR doc tree diagram.

        Candidate tree:              Session tree:

              A                      Session 1 (resume):  A->B -> B->C -> C->D
             / \\                     Session 2 (fork@B):  A->B -> B->E
            B   G                    Session 3 (create):  A->G
           / \\
          C   E
          |
          D
        """
        factory, _ = _make_echo_factory()

        # --- Session 1: resume through A->B->C->D ---
        mgr1 = SessionManager(create=factory, strategy=AlwaysResume())

        session1 = mgr1.select(parent_candidate_idx=-1)
        session1.resume("mutate A")  # -> B
        mgr1.checkpoint(candidate_idx=0, accepted=True)

        session1 = mgr1.select(parent_candidate_idx=0)
        assert session1 is mgr1.current_session()  # same live session (resume)
        session1.resume("mutate B")  # -> C
        mgr1.checkpoint(candidate_idx=1, accepted=True)

        session1.resume("mutate C")  # -> D
        mgr1.checkpoint(candidate_idx=2, accepted=True)

        assert len(session1.history) == 6  # 3 resume x 2 messages

        # --- Session 2: fork from checkpoint[0] (state at B) ---
        # Simulate: a different strategy forks from B's checkpoint
        checkpoint_b = mgr1.checkpoints[0]
        session2 = checkpoint_b.fork()
        assert len(session2.history) == 2  # only A->B
        session2.resume("mutate B differently")  # -> E
        assert len(session2.history) == 4

        # Checkpoint at B is untouched
        assert len(mgr1.checkpoints[0].history) == 2

        # --- Session 3: fresh (create) ---
        mgr3 = SessionManager(create=factory, strategy=AlwaysCreate())
        session3 = mgr3.select(parent_candidate_idx=-1)
        assert len(session3.history) == 0
        session3.resume("mutate A fresh")  # -> G
        assert len(session3.history) == 2

        # All three sessions are independent
        assert len(session1.history) == 6
        assert len(session2.history) == 4
        assert len(session3.history) == 2
