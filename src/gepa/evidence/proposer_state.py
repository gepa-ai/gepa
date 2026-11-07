"""Checkpoint the audited native sampler and its shared RNG relationships."""

from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from typing import Any


def proposer_signature(proposer: Any) -> dict[str, Any]:
    selector = proposer.candidate_selector
    sampling = proposer.sampling_strategy
    return {
        "candidate_selector": type(selector).__name__,
        "epsilon": getattr(selector, "epsilon", None),
        "top_k": getattr(selector, "k", None),
        "sampling_strategy": type(sampling).__name__,
        "parents": getattr(sampling, "p", None),
        "mutations": getattr(sampling, "n", None),
        "module_selector": type(proposer.module_selector).__name__,
        "batch_sampler": type(proposer.batch_sampler).__name__,
        "minibatch_size": proposer.batch_sampler.minibatch_size,
        "perfect_score": proposer.perfect_score,
        "skip_perfect_score": proposer.skip_perfect_score,
    }


def _rng_roles(proposer: Any) -> dict[str, random.Random]:
    roles = {"sampler": proposer.batch_sampler.rng}
    candidate_rng = getattr(proposer.candidate_selector, "rng", None)
    if candidate_rng is not None:
        roles["candidate"] = candidate_rng
    if any(type(rng) is not random.Random for rng in roles.values()):
        raise ValueError("evidence checkpoints require native random.Random sources")
    return roles


def snapshot_proposer(proposer: Any) -> dict[str, Any]:
    sampler = proposer.batch_sampler
    groups: list[dict[str, Any]] = []
    identities: dict[int, int] = {}
    for role, rng in _rng_roles(proposer).items():
        group_index = identities.get(id(rng))
        if group_index is None:
            identities[id(rng)] = len(groups)
            groups.append({"roles": [role], "state": rng.getstate()})
        else:
            groups[group_index]["roles"].append(role)
    return {
        "schema_version": 1,
        "signature": proposer_signature(proposer),
        "rng_groups": groups,
        "sampler": {
            "shuffled_ids": list(sampler.shuffled_ids),
            "epoch": sampler.epoch,
            "id_freqs": list(sampler.id_freqs.items()),
            "last_trainset_size": sampler.last_trainset_size,
            "current_iteration": sampler._current_iteration,
            "calls_in_iteration": sampler._calls_in_iteration,
        },
    }


def _integer(value: Any, *, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def restore_proposer(proposer: Any, saved: dict[str, Any], available_ids: list[Any]) -> None:
    """Validate the entire checkpoint before mutating any live native strategy."""
    if saved.get("schema_version") != 1 or saved.get("signature") != proposer_signature(proposer):
        raise ValueError("native proposer configuration changed across evidence resume")
    roles = _rng_roles(proposer)
    expected_aliases = {frozenset(role for role, peer in roles.items() if peer is rng) for rng in roles.values()}
    validated_groups = []
    observed_roles: set[str] = set()
    observed_aliases = set()
    try:
        for group in saved["rng_groups"]:
            names = group["roles"]
            if (
                not names
                or len(set(names)) != len(names)
                or set(names) - roles.keys()
                or observed_roles.intersection(names)
            ):
                raise ValueError("invalid saved native RNG roles")
            observed_roles.update(names)
            observed_aliases.add(frozenset(names))
            probe = random.Random()
            probe.setstate(group["state"])
            validated_groups.append((names, probe.getstate()))
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("corrupt native RNG checkpoint") from error
    if observed_roles != set(roles) or observed_aliases != expected_aliases:
        raise ValueError("native RNG sharing changed across evidence resume")
    try:
        sampler = saved["sampler"]
        shuffled = list(sampler["shuffled_ids"])
        frequencies = dict(sampler["id_freqs"])
        if len(frequencies) != len(sampler["id_freqs"]) or any(
            not _integer(v, minimum=1) for v in frequencies.values()
        ):
            raise ValueError("invalid saved sampler frequencies")
        if not _integer(sampler["epoch"], minimum=-1) or not _integer(sampler["last_trainset_size"], minimum=0):
            raise ValueError("invalid saved sampler epoch or size")
        current_iteration = sampler["current_iteration"]
        if current_iteration is not None and not _integer(current_iteration, minimum=0):
            raise ValueError("invalid saved sampler iteration")
        if not _integer(sampler["calls_in_iteration"], minimum=0):
            raise ValueError("invalid saved sampler call offset")
        if frequencies != Counter(shuffled):
            raise ValueError("sampler frequencies differ from saved shuffled IDs")
        if shuffled:
            if set(shuffled) != set(available_ids) or len(shuffled) % proposer.batch_sampler.minibatch_size:
                raise ValueError("saved sampler IDs differ from the fixed training context")
            if sampler["last_trainset_size"] != len(available_ids) or sampler["epoch"] < 0:
                raise ValueError("saved sampler size differs from fixed training context")
        elif sampler["epoch"] != -1 or sampler["last_trainset_size"] != 0 or frequencies:
            raise ValueError("invalid uninitialized sampler checkpoint")
    except (KeyError, TypeError) as error:
        raise ValueError("corrupt native sampler checkpoint") from error
    for names, state in validated_groups:
        roles[names[0]].setstate(state)
    target = proposer.batch_sampler
    target.shuffled_ids = deepcopy(shuffled)
    target.id_freqs = Counter(frequencies)
    target.epoch = sampler["epoch"]
    target.last_trainset_size = sampler["last_trainset_size"]
    target._current_iteration = sampler["current_iteration"]
    target._calls_in_iteration = sampler["calls_in_iteration"]
