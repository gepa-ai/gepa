# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Agent-aware proposer protocol for session-based candidate generation.

An ``AgentProposer`` composes a ``Session`` (interaction context) with a
``WorkspaceManager`` (isolated code workspaces) to produce candidates via
coding agents or LLMs.

The engine calls ``propose(state)`` exactly as with any other
``ProposeNewCandidate`` implementation — it does not know whether the proposer
is session-based or not.

Typical internal flow::

    1. Select parent candidate from state (e.g. Pareto front)
    2. Fork parent's workspace  → isolated directory for the agent
    3. Fork parent's session    → preserved conversation context
    4. Send mutation instruction to session (agent modifies files)
    5. Snapshot workspace state
    6. Return CandidateProposal with {"workspace": new_ref}
"""

from __future__ import annotations

from typing import Any, Protocol

from gepa.core.data_loader import DataId
from gepa.core.session import Session
from gepa.core.state import GEPAState
from gepa.core.workspace import WorkspaceManager
from gepa.proposer.base import CandidateProposal


class AgentProposer(Protocol[DataId]):
    """Proposer that uses a Session and WorkspaceManager to generate candidates.

    Concrete implementations (e.g. ``ClaudeCodeProposer``, ``CodexProposer``)
    live in ``src/gepa/core/sessions/`` or ``src/gepa/core/proposers/`` and
    are wired by the adapter layer (e.g. ``optimize_anything``).

    The protocol is intentionally minimal — it only requires ``propose()``.
    How sessions and workspaces are managed internally is up to the
    implementation.
    """

    session: Session
    workspace_manager: WorkspaceManager

    def propose(self, state: GEPAState[Any, DataId]) -> CandidateProposal[DataId] | None:
        """Propose a new candidate or return ``None`` if no proposal is made.

        Implementations should:
        1. Select a parent candidate from ``state``
        2. Fork workspace and session
        3. Send mutation instruction to the session
        4. Return a ``CandidateProposal`` with the new candidate dict
        """
        ...
