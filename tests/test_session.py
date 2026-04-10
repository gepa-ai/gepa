# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session -- Session protocol, strategies, and manager."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from gepa.core.session import (
    LLMSession,
    NullSession,
    Session,
)
from gepa.core.session_manager import (
    SessionContext,
    SessionRecord,
    SessionManager,
    SessionStrategy,
    make_session_lm,
    resolve_session_strategy,
)
from gepa.strategies.session_strategy import (
    AlwaysFork,
    AlwaysReset,
    ParentLinked,
    RandomStrategy,
    RoundRobin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_session(system_prompt: str = "test", session_id: str | None = None) -> LLMSession:
    """Create a session with a simple echo API call."""

    def echo_api(messages: list[dict], **kwargs) -> str:  # type: ignore[type-arg]
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    return LLMSession(system_prompt=system_prompt, api_call=echo_api, session_id=session_id)


def _make_echo_factory(system_prompt: str = "test") -> tuple[Callable[[], LLMSession], Callable[..., str]]:
    """Return (factory, echo_api) for building test sessions."""

    def echo_api(messages: list[dict], **kwargs) -> str:  # type: ignore[type-arg]
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def factory() -> LLMSession:
        return LLMSession(system_prompt=system_prompt, api_call=echo_api)

    return factory, echo_api


# ===========================================================================
# LLMSession
# ===========================================================================


class TestLLMSession:
    def test_protocol_compliance(self):
        session = _make_echo_session()
        assert isinstance(session, Session)

    def test_send_appends_messages(self):
        session = _make_echo_session()
        result = session.send("hello")
        assert result == "echo: hello"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "hello"}
        assert session.history[1] == {"role": "assistant", "content": "echo: hello"}

    def test_send_multiple(self):
        session = _make_echo_session()
        session.send("a")
        session.send("b")
        assert len(session.history) == 4

    def test_fork_copies_history(self):
        session = _make_echo_session()
        session.send("hello")
        forked = session.fork()
        assert forked.history == session.history
        assert forked.session_id != session.session_id

    def test_fork_diverges(self):
        session = _make_echo_session()
        session.send("hello")
        forked = session.fork()
        forked.send("world")
        assert len(forked.history) == 4
        assert len(session.history) == 2

    def test_reset_creates_empty_session(self):
        session = _make_echo_session()
        session.send("hello")
        new = session.reset()
        assert new.history == []
        assert new.session_id != session.session_id

    def test_reset_preserves_backend(self):
        session = _make_echo_session(system_prompt="custom")
        new = session.reset()
        result = new.send("test")
        assert result == "echo: test"

    def test_history_is_copy(self):
        session = _make_echo_session()
        session.send("hello")
        h = session.history
        h.append({"role": "user", "content": "injected"})
        assert len(session.history) == 2

    def test_session_id_auto_generated(self):
        s1 = _make_echo_session()
        s2 = _make_echo_session()
        assert s1.session_id != s2.session_id

    def test_session_id_explicit(self):
        session = _make_echo_session(session_id="my-id")
        assert session.session_id == "my-id"

    def test_api_call_receives_system_prompt(self):
        received = []

        def spy_api(messages, **kwargs):
            received.append(messages)
            return "ok"

        session = LLMSession(system_prompt="sys", api_call=spy_api)
        session.send("hello")
        assert received[0][0] == {"role": "system", "content": "sys"}


# ===========================================================================
# NullSession
# ===========================================================================


class TestNullSession:
    def test_protocol_compliance(self):
        assert isinstance(NullSession(), Session)

    def test_send_returns_empty(self):
        assert NullSession().send("hello") == ""

    def test_history_empty(self):
        assert NullSession().history == []

    def test_fork_returns_new_null(self):
        s = NullSession()
        f = s.fork()
        assert isinstance(f, NullSession)
        assert f.session_id != s.session_id

    def test_reset_returns_new_null(self):
        s = NullSession()
        r = s.reset()
        assert isinstance(r, NullSession)
        assert r.session_id != s.session_id


# ===========================================================================
# make_session_lm
# ===========================================================================


class TestMakeSessionLm:
    def test_string_prompt(self):
        session = _make_echo_session()
        lm = make_session_lm(session)
        assert lm("hello") == "echo: hello"

    def test_message_list_prompt(self):
        session = _make_echo_session()
        lm = make_session_lm(session)
        result = lm([{"role": "user", "content": "hello"}])
        assert result == "echo: hello"

    def test_session_accumulates_history(self):
        session = _make_echo_session()
        lm = make_session_lm(session)
        lm("a")
        lm("b")
        assert len(session.history) == 4

    def test_callable_satisfies_lm_protocol(self):
        session = _make_echo_session()
        lm = make_session_lm(session)
        assert callable(lm)

    def test_dynamic_provider(self):
        factory, _ = _make_echo_factory()
        sessions: list[LLMSession] = []

        def provider():
            s = factory()
            sessions.append(s)
            return s

        lm = make_session_lm(provider)
        lm("hello")
        assert len(sessions) == 1


# ===========================================================================
# SessionStrategy — protocol conformance
# ===========================================================================


class TestSessionStrategy:
    def test_protocol_compliance(self):
        assert isinstance(AlwaysFork(), SessionStrategy)
        assert isinstance(AlwaysReset(), SessionStrategy)
        assert isinstance(RandomStrategy(), SessionStrategy)
        assert isinstance(RoundRobin(), SessionStrategy)
        assert isinstance(ParentLinked(), SessionStrategy)


def _ctx(
    sessions: dict | None = None,
    *,
    parent_candidate_idx: int | None = None,
    iteration: int = 0,
    create: Callable[[], Session] | None = None,
) -> SessionContext:
    factory, _ = _make_echo_factory()
    return SessionContext(
        parent_candidate_idx=parent_candidate_idx,
        iteration=iteration,
        sessions=sessions or {},
        create=create or factory,
    )


# ===========================================================================
# Built-in strategies
# ===========================================================================


class TestAlwaysFork:
    def test_empty_sessions_calls_create(self):
        session = AlwaysFork().select(_ctx())
        assert isinstance(session, LLMSession)
        assert session.history == []

    def test_forks_most_recent(self):
        s1 = _make_echo_session()
        s1.send("hello")
        store = {0: SessionRecord(s1)}
        forked = AlwaysFork().select(_ctx(store))
        assert forked.history == s1.history
        assert forked.session_id != s1.session_id

    def test_observe_binds_on_accept(self):
        """Accepted candidates are stored in the session map via SessionManager."""
        manager = SessionManager(create=_make_echo_session, strategy=AlwaysFork())
        session = manager.select()
        manager.observe(candidate_idx=7, accepted=True, val_score=0.8)
        assert 7 in manager.sessions
        assert manager.sessions[7].session is session
        assert manager.sessions[7].val_score == 0.8

    def test_observe_skips_on_reject(self):
        """Rejected candidates are not stored."""
        manager = SessionManager(create=_make_echo_session, strategy=AlwaysFork())
        manager.select()
        manager.observe(candidate_idx=3, accepted=False)
        assert 3 not in manager.sessions

    def test_observe_skips_if_no_candidate_idx(self):
        """Candidates with no idx are not stored."""
        manager = SessionManager(create=_make_echo_session, strategy=AlwaysFork())
        manager.select()
        manager.observe(candidate_idx=None, accepted=True)
        assert len(manager.sessions) == 0


class TestAlwaysReset:
    def test_resets_most_recent(self):
        s1 = _make_echo_session()
        s1.send("hello")
        store = {0: SessionRecord(s1)}
        reset = AlwaysReset().select(_ctx(store))
        assert reset.history == []
        assert reset.session_id != s1.session_id

    def test_empty_calls_create(self):
        session = AlwaysReset().select(_ctx())
        assert isinstance(session, LLMSession)


class TestRandomStrategy:
    def test_always_fork(self):
        s1 = _make_echo_session()
        s1.send("hello")
        store = {0: SessionRecord(s1)}
        result = RandomStrategy(fork_probability=1.0).select(_ctx(store))
        assert result.history == s1.history

    def test_always_reset(self):
        s1 = _make_echo_session()
        s1.send("hello")
        store = {0: SessionRecord(s1)}
        result = RandomStrategy(fork_probability=0.0).select(_ctx(store))
        assert result.history == []


class TestRoundRobin:
    def test_alternates_fork_reset(self):
        s1 = _make_echo_session()
        s1.send("hello")
        store = {0: SessionRecord(s1)}
        rr = RoundRobin()
        r1 = rr.select(_ctx(store))  # fork
        r2 = rr.select(_ctx(store))  # reset
        r3 = rr.select(_ctx(store))  # fork
        assert len(r1.history) == 2
        assert len(r2.history) == 0
        assert len(r3.history) == 2


class TestParentLinked:
    def test_forks_from_parent(self):
        parent = _make_echo_session()
        parent.send("parent-msg")
        other = _make_echo_session()
        other.send("other")
        store = {
            5: SessionRecord(parent),
            6: SessionRecord(other),
        }
        result = ParentLinked().select(_ctx(store, parent_candidate_idx=5))
        assert result.history == parent.history
        assert result.session_id != parent.session_id

    def test_falls_back_to_most_recent_when_parent_unknown(self):
        recent = _make_echo_session()
        recent.send("recent")
        store = {0: SessionRecord(recent)}
        # parent_candidate_idx=99 not in store → fall back to most recent
        result = ParentLinked().select(_ctx(store, parent_candidate_idx=99))
        assert result.history == recent.history

    def test_empty_store_calls_create(self):
        result = ParentLinked().select(_ctx(parent_candidate_idx=5))
        assert isinstance(result, LLMSession)

    def test_sibling_mutations_are_independent(self):
        parent = _make_echo_session()
        parent.send("A->B")
        store = {0: SessionRecord(parent)}
        strategy = ParentLinked()
        ctx = _ctx(store, parent_candidate_idx=0)
        child1 = strategy.select(ctx)
        child2 = strategy.select(ctx)
        child1.send("child1-only")
        child2.send("child2-only")
        # Siblings don't interfere; parent untouched
        assert len(parent.history) == 2
        assert len(child1.history) == 4
        assert len(child2.history) == 4
        assert child1.history[-1]["content"] == "echo: child1-only"
        assert child2.history[-1]["content"] == "echo: child2-only"


# ===========================================================================
# SessionManager — keyed store + strategy delegation
# ===========================================================================


class TestSessionManager:
    def test_default_strategy_is_fork(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        assert isinstance(manager._strategy, AlwaysFork)

    def test_select_with_fork_strategy(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        s1 = manager.select()
        s1.send("first")
        manager.observe(candidate_idx=0, accepted=True, val_score=0.5)

        # Fork gives a new session seeded from s1's history
        s2 = manager.select()
        assert s2 is not s1
        assert s2.history == s1.history

    def test_select_with_reset_strategy(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysReset())

        s1 = manager.select()
        s1.send("hello")
        manager.observe(candidate_idx=0, accepted=True)

        s2 = manager.select()
        assert s2 is not s1
        assert s2.history == []

    def test_observe_merges_into_store(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        manager.select()
        manager.observe(candidate_idx=0, accepted=True, val_score=0.42)

        assert 0 in manager.sessions
        assert manager.sessions[0].val_score == 0.42

    def test_observe_on_reject_does_not_bind(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        manager.select()
        manager.observe(candidate_idx=0, accepted=False)

        assert 0 not in manager.sessions

    def test_iteration_counter_increments(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        assert manager.iteration == 0
        manager.select()
        assert manager.iteration == 1
        manager.select()
        assert manager.iteration == 2

    def test_sessions_view_is_readonly_snapshot(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        manager.select()
        manager.observe(candidate_idx=0, accepted=True)
        view = manager.sessions
        # Mutating the returned view doesn't affect the manager
        view.clear()  # type: ignore[attr-defined]
        assert 0 in manager.sessions

    def test_current_session_creates_if_none(self):
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory)
        s = manager.current_session()
        assert isinstance(s, LLMSession)

    def test_parent_linked_scenario(self):
        """Full candidate-tree scenario using ParentLinked strategy.

        Candidate tree:   A -> B -> C
                              \\-> D
        """
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=ParentLinked())

        # Iter 1: root mutation -> candidate A (idx 0)
        s_a = manager.select(parent_candidate_idx=None)
        s_a.send("mutate root")
        manager.observe(candidate_idx=0, accepted=True, val_score=0.3)

        # Iter 2: mutate A -> candidate B (idx 1), fork from A
        s_b = manager.select(parent_candidate_idx=0)
        s_b.send("mutate A")
        manager.observe(candidate_idx=1, accepted=True, val_score=0.6)

        # Iter 3: mutate B -> candidate C (idx 2), fork from B
        s_c = manager.select(parent_candidate_idx=1)
        s_c.send("mutate B")
        manager.observe(candidate_idx=2, accepted=True, val_score=0.8)

        # Iter 4: mutate B again -> candidate D (idx 3), fork from B (sibling of C)
        s_d = manager.select(parent_candidate_idx=1)
        s_d.send("mutate B sibling")
        manager.observe(candidate_idx=3, accepted=True, val_score=0.7)

        # All four candidates bound with their scores
        assert set(manager.sessions.keys()) == {0, 1, 2, 3}
        assert manager.sessions[2].val_score == 0.8
        assert manager.sessions[3].val_score == 0.7

        # C and D both fork from B's session -> both inherit [root, A, B] history
        # (6 messages each after their own send).  Parent B is untouched.
        assert len(s_c.history) == 6
        assert len(s_d.history) == 6
        assert len(s_b.history) == 4

    def test_val_score_tracking_across_iterations(self):
        """Scores flow from observe() into the sessions store and can be read back."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        manager.select()
        manager.observe(candidate_idx=0, accepted=True, val_score=0.5)
        manager.select()
        manager.observe(candidate_idx=1, accepted=True, val_score=0.9)
        manager.select()
        manager.observe(candidate_idx=2, accepted=True, val_score=0.3)

        best_idx = max(manager.sessions, key=lambda k: manager.sessions[k].val_score or 0)
        assert best_idx == 1

    def test_custom_strategy(self):
        select_count = 0

        class CountingStrategy:
            def select(self, ctx: SessionContext) -> Session:
                nonlocal select_count
                select_count += 1
                return ctx.create()

        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=CountingStrategy())
        manager.select()
        manager.observe(candidate_idx=0, accepted=True)
        manager.select()
        manager.observe(candidate_idx=1, accepted=False)
        assert select_count == 2
        assert 0 in manager.sessions  # accepted
        assert 1 not in manager.sessions  # rejected


# ===========================================================================
# Resolver
# ===========================================================================


class TestResolveSessionStrategy:
    def test_string_fork(self):
        assert isinstance(resolve_session_strategy("fork"), AlwaysFork)

    def test_string_reset(self):
        assert isinstance(resolve_session_strategy("reset"), AlwaysReset)

    def test_string_random(self):
        assert isinstance(resolve_session_strategy("random"), RandomStrategy)

    def test_string_round_robin(self):
        assert isinstance(resolve_session_strategy("round_robin"), RoundRobin)

    def test_string_parent_linked(self):
        assert isinstance(resolve_session_strategy("parent_linked"), ParentLinked)

    def test_passthrough(self):
        strategy = AlwaysFork()
        assert resolve_session_strategy(strategy) is strategy

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown session_strategy"):
            resolve_session_strategy("nonexistent")


# ===========================================================================
# End-to-end mock: LLM API optimization loop with sessions
# ===========================================================================


class TestEndToEndLLMSession:
    """Simulate a full optimization loop with LLMSession + SessionManager.

    This exercises the complete data flow that the engine performs:
    select(parent) → session.send(prompt) → evaluate → observe(idx, accepted, score)
    repeated across multiple iterations, verifying that session state, history,
    val_scores, and lineage tracking all work correctly.
    """

    def test_fork_strategy_accumulates_history(self):
        """AlwaysFork: each accepted mutation's session carries forward full history."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        # Iteration 0: root mutation
        s0 = manager.select(parent_candidate_idx=None)
        response_0 = s0.send("Improve the initial prompt")
        assert response_0 == "echo: Improve the initial prompt"
        manager.observe(candidate_idx=0, accepted=True, val_score=0.3)

        # Iteration 1: mutate candidate 0 → accepted as candidate 1
        s1 = manager.select(parent_candidate_idx=0)
        response_1 = s1.send("Make it more concise")
        assert response_1 == "echo: Make it more concise"
        # s1 was forked from most-recent → carries s0's history
        assert len(s1.history) == 4  # 2 from s0 + 2 from this send
        manager.observe(candidate_idx=1, accepted=True, val_score=0.6)

        # Iteration 2: mutate candidate 1 → rejected
        s2 = manager.select(parent_candidate_idx=1)
        s2.send("Add examples")
        manager.observe(candidate_idx=2, accepted=False, val_score=0.2)
        # Rejected → not bound in store
        assert 2 not in manager.sessions

        # Iteration 3: mutate candidate 1 again → accepted
        s3 = manager.select(parent_candidate_idx=1)
        s3.send("Use chain of thought")
        manager.observe(candidate_idx=3, accepted=True, val_score=0.8)

        # Final state: 3 accepted candidates, scores tracked
        assert set(manager.sessions.keys()) == {0, 1, 3}
        assert manager.sessions[0].val_score == 0.3
        assert manager.sessions[1].val_score == 0.6
        assert manager.sessions[3].val_score == 0.8
        assert manager.iteration == 4

    def test_reset_strategy_stateless_behavior(self):
        """AlwaysReset: each iteration starts with empty history (current GEPA default)."""
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysReset())

        # Iteration 0
        s0 = manager.select()
        s0.send("First prompt")
        manager.observe(candidate_idx=0, accepted=True, val_score=0.5)

        # Iteration 1: reset gives clean slate
        s1 = manager.select()
        assert s1.history == []  # no history from previous iteration
        s1.send("Second prompt")
        assert len(s1.history) == 2  # only this iteration's messages
        manager.observe(candidate_idx=1, accepted=True, val_score=0.7)

    def test_parent_linked_lineage(self):
        """ParentLinked: forking from parent builds a session tree matching the candidate tree.

        Candidate tree:
              0 (root)
             / \\
            1   3
            |
            2

        Session tree should mirror this: s1 forks from s0, s2 from s1, s3 from s0.
        """
        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=ParentLinked())

        # Root: create session, produce candidate 0
        s0 = manager.select(parent_candidate_idx=None)
        s0.send("root mutation")
        manager.observe(candidate_idx=0, accepted=True, val_score=0.3)

        # Mutate 0 → candidate 1
        s1 = manager.select(parent_candidate_idx=0)
        s1.send("mutate 0 → 1")
        manager.observe(candidate_idx=1, accepted=True, val_score=0.5)
        # s1 should have root's history + its own
        assert len(s1.history) == 4

        # Mutate 1 → candidate 2
        s2 = manager.select(parent_candidate_idx=1)
        s2.send("mutate 1 → 2")
        manager.observe(candidate_idx=2, accepted=True, val_score=0.8)
        # s2 should have root + 0→1 + 1→2
        assert len(s2.history) == 6

        # Mutate 0 again → candidate 3 (sibling of 1)
        s3 = manager.select(parent_candidate_idx=0)
        s3.send("mutate 0 → 3")
        manager.observe(candidate_idx=3, accepted=True, val_score=0.4)
        # s3 forks from s0 → only root history + its own
        assert len(s3.history) == 4

        # s0 untouched by any fork
        assert len(s0.history) == 2

        # All scores tracked
        assert manager.sessions[2].val_score == 0.8
        best = max(manager.sessions, key=lambda k: manager.sessions[k].val_score or 0)
        assert best == 2

    def test_dynamic_lm_routes_through_session(self):
        """make_session_lm + SessionManager: the proposer's lm() call hits the current session."""
        from gepa.core.session_manager import make_session_lm

        factory, _ = _make_echo_factory()
        manager = SessionManager(create=factory, strategy=AlwaysFork())

        # Wire dynamic LM the same way api.py does
        dynamic_lm = make_session_lm(manager.current_session)

        # Iteration 0
        manager.select()
        result = dynamic_lm("Improve this prompt")  # goes through current_session.send()
        assert result == "echo: Improve this prompt"
        manager.observe(candidate_idx=0, accepted=True, val_score=0.5)

        # Iteration 1: select new session, dynamic_lm automatically uses it
        manager.select()
        result = dynamic_lm("Make it better")
        assert result == "echo: Make it better"
        # Current session has forked history from iter 0 + new message
        current = manager.current_session()
        assert len(current.history) == 4


# ===========================================================================
# Integration: gepa.optimize() with session_strategy
# ===========================================================================


class TestOptimizeWithSessionStrategy:
    """Call gepa.optimize() with session_strategy to verify full wiring.

    Uses a MinimalAdapter that forces the reflective-mutation path (no
    propose_new_texts), and a deterministic mock LM.  The primary assertion
    is that the optimization completes without error — proving that api.py's
    session wiring, the engine's select()/observe() calls, and the proposer's
    dynamic_lm all compose correctly.
    """

    @staticmethod
    def _make_minimal_adapter():
        """Adapter with evaluate + make_reflective_dataset only (no propose_new_texts)."""
        from gepa.core.adapter import EvaluationBatch

        class MinimalAdapter:
            def evaluate(self, batch, candidate, capture_traces=False):
                # Score based on prompt length — longer is better (deterministic)
                text = candidate.get("prompt", "")
                score = min(len(text) / 200, 1.0)
                outputs = [{"text": text}] * len(batch)
                scores = [score] * len(batch)
                trajectories = [{"score": s} for s in scores] if capture_traces else None
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                return {c: [{"feedback": "make it longer"}] for c in components_to_update}

        return MinimalAdapter()

    @staticmethod
    def _make_mock_lm():
        """Mock LM that returns a fenced code block with an improved prompt."""
        call_count = [0]

        def mock_lm(prompt):
            call_count[0] += 1
            return f"```\nThis is an improved prompt version {call_count[0]} that is deliberately longer to score higher in our length-based evaluation metric.\n```"

        return mock_lm

    def test_optimize_with_parent_linked_strategy(self):
        """gepa.optimize() with session_strategy='parent_linked' completes.

        Verifies that parent_candidate_idx flows from the proposer's
        candidate selection through to SessionStrategy.select(ctx).
        """
        import gepa

        result = gepa.optimize(
            seed_candidate={"prompt": "initial short prompt"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._make_minimal_adapter(),
            max_metric_calls=30,
            reflection_lm=self._make_mock_lm(),
            session_strategy="parent_linked",
        )
        assert result is not None
        assert result.total_metric_calls > 0

    def test_optimize_with_fork_strategy(self):
        """gepa.optimize() with session_strategy='fork' completes without error."""
        import gepa

        result = gepa.optimize(
            seed_candidate={"prompt": "initial short prompt"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._make_minimal_adapter(),
            max_metric_calls=30,
            reflection_lm=self._make_mock_lm(),
            session_strategy="fork",
        )
        assert result is not None
        assert result.total_metric_calls > 0

    def test_optimize_with_reset_strategy(self):
        """gepa.optimize() with session_strategy='reset' (stateless) completes."""
        import gepa

        result = gepa.optimize(
            seed_candidate={"prompt": "initial short prompt"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._make_minimal_adapter(),
            max_metric_calls=30,
            reflection_lm=self._make_mock_lm(),
            session_strategy="reset",
        )
        assert result is not None
        assert result.total_metric_calls > 0

    def test_optimize_with_best_score_strategy(self):
        """gepa.optimize() with session_strategy='best_score' completes."""
        import gepa

        result = gepa.optimize(
            seed_candidate={"prompt": "initial short prompt"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._make_minimal_adapter(),
            max_metric_calls=30,
            reflection_lm=self._make_mock_lm(),
            session_strategy="best_score",
        )
        assert result is not None
        assert result.total_metric_calls > 0

    def test_optimize_without_session_strategy_unchanged(self):
        """session_strategy=None (default) still works — backward compatibility."""
        import gepa

        result = gepa.optimize(
            seed_candidate={"prompt": "initial short prompt"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._make_minimal_adapter(),
            max_metric_calls=30,
            reflection_lm=self._make_mock_lm(),
        )
        assert result is not None
        assert result.total_metric_calls > 0
