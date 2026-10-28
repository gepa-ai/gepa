"""Curator owns provenance; adapters only see their original evaluation interface."""

from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.task_curator import TaskCuratorContractError, TaskCuratorSampler


class NamedLoader:
    def all_ids(self):
        return ["task-A", "task-B"]

    def fetch(self, ids):
        return [{"label": tid} for tid in ids]

    def __len__(self):
        return 2


@pytest.fixture
def anchored():
    sampler = TaskCuratorSampler(
        minibatch_size=2, lm=Mock(return_value='{"decision":"run","selected_success":"before"}')
    )
    state = SimpleNamespace(i=0, full_program_trace=[{"i": 0}])
    for score in (1, 0):
        ids = sampler.next_minibatch_ids_for_candidate(NamedLoader(), state, candidate_idx=0)
        batch = EvaluationBatch(
            outputs=[{"output": tid} for tid in ids],
            scores=[score] * len(ids),
            trajectories=[{"task": tid} for tid in ids],
            objective_scores=[{"quality": i} for i in range(len(ids))],
            num_metric_calls=7,
        )
        sampler.observe_evaluation(iteration=state.i + 1, task_ids=ids, evaluation=batch, state=state)
        if score == 1:
            state.i += 1
            state.full_program_trace.append({"i": state.i})
    return sampler, state, ids, batch


def build(anchored, builder, *, components=("p", "q")):
    sampler, state, ids, batch = anchored
    return sampler.build_reflective_dataset(
        candidate=dict.fromkeys(components, "prompt"),
        task_ids=ids,
        evaluation=batch,
        components_to_update=list(components),
        make_reflective_dataset=builder,
        state=state,
    )


@pytest.mark.parametrize("layout", ["reordered", "filtered", "repeated", "empty", "omitted_component"])
def test_transformed_records_keep_their_own_task_anchor(anchored, layout):
    sampler, state, ids, batch = anchored
    returned = []
    seen = []
    original = deepcopy(batch)

    def builder(candidate, single, components):
        assert not hasattr(single, "task_ids")
        assert len(single.scores) == len(single.outputs) == len(single.trajectories) == 1
        assert single.num_metric_calls is None
        tid = single.trajectories[0]["task"]
        i = ids.index(tid)
        assert single.outputs == [batch.outputs[i]]
        assert single.objective_scores == [batch.objective_scores[i]]
        seen.append(tid)
        rows = [{"owner": tid, "step": step} for step in range(3)]
        if layout == "reordered":
            rows.reverse()
        elif layout == "filtered":
            rows = rows[1:] if tid == ids[1] else []
        elif layout == "repeated":
            rows = [rows[1], *rows, rows[1]]
        elif layout == "empty":
            rows = []
        dataset = {"p": rows, "q": rows[::-1]}
        if layout == "omitted_component":
            del dataset["p" if tid == ids[0] else "q"]
        returned.append(dataset)
        return dataset

    result = build(anchored, builder)
    assert seen == ids
    expected = {
        component: [record for dataset in returned for record in dataset.get(component, [])] for component in result
    }
    for component, records in result.items():
        assert [(r["owner"], r["step"]) for r in records] == [(r["owner"], r["step"]) for r in expected[component]]
        for record in records:
            assert record["HistoricalSuccess"]["successful_state"]["task"] == record["owner"]
            assert "_gepa_task_id" not in record
    assert all("HistoricalSuccess" not in r for d in returned for rs in d.values() for r in rs)
    assert batch == original
    audited = state.full_program_trace[-1]["task_curator_reflection_anchors"]
    expected_ids = {r["owner"] for rows in expected.values() for r in rows}
    assert [row["task_id"] for row in audited] == [tid for tid in ids if tid in expected_ids]


def test_shared_adapter_buffers_and_fake_id_cannot_corrupt_identity(anchored):
    _, _, ids, _ = anchored
    shared = {"Feedback": {"owner": None}, "_gepa_task_id": "untrusted"}

    def builder(candidate, single, components):
        shared["Feedback"]["owner"] = single.trajectories[0]["task"]
        return dict.fromkeys(components, [shared])  # noqa: RUF024 -- exercise aliased adapter buffers

    result = build(anchored, builder)
    for records in result.values():
        assert [r["Feedback"]["owner"] for r in records] == ids
        for record in records:
            assert record["HistoricalSuccess"]["successful_state"]["task"] == record["Feedback"]["owner"]
            # An adapter-owned field is preserved as evidence, never interpreted as identity.
            assert record["_gepa_task_id"] == "untrusted"
    assert "HistoricalSuccess" not in shared
    result["p"][0]["HistoricalSuccess"]["successful_state"]["task"] = "changed"
    assert result["q"][0]["HistoricalSuccess"]["successful_state"]["task"] == ids[0]


def test_current_success_does_not_get_a_failure_anchor(anchored):
    sampler, state, ids, batch = anchored
    result = build(
        (sampler, state, ids, replace(batch, scores=[1, 0])),
        lambda _, single, components: {c: [{"owner": single.trajectories[0]["task"]}] for c in components},
    )
    for records in result.values():
        assert "HistoricalSuccess" not in records[0]
        assert records[1]["HistoricalSuccess"]["successful_state"]["task"] == ids[1]


@pytest.mark.parametrize("field", ["scores", "outputs", "trajectories", "objective_scores", "ids"])
def test_bad_alignment_fails_before_adapter_call(anchored, field):
    sampler, state, ids, batch = anchored
    if field == "ids":
        ids = [ids[0], ids[0]]
    else:
        batch = replace(batch, **{field: []})
    builder = Mock()
    with pytest.raises(TaskCuratorContractError, match="do not align|must be unique"):
        build((sampler, state, ids, batch), builder)
    builder.assert_not_called()


@pytest.mark.parametrize("error_type", [Exception, RuntimeError, ValueError])
def test_singleton_exception_is_explicit_not_empty_success_or_batch_fallback(anchored, error_type):
    _, state, ids, _ = anchored
    error = error_type("No valid predictions found for any module.")
    builder = Mock(side_effect=[{"p": [{}]}, error])
    with pytest.raises(TaskCuratorContractError, match="Cannot build single-task reflection") as caught:
        build(anchored, builder)
    assert ids[1] in str(caught.value)
    assert caught.value.__cause__ is error
    assert builder.call_count == 2
    assert "task_curator_reflection_anchors" not in state.full_program_trace[-1]


def test_anchors_disabled_preserve_whole_batch_builder_semantics(anchored):
    sampler, _, _, batch = anchored
    sampler.use_reflection_anchors = False
    dataset = {"p": [{"Feedback": "joint evidence"}]}
    builder = Mock(return_value=dataset)
    assert build(anchored, builder) is dataset
    builder.assert_called_once_with({"p": "prompt", "q": "prompt"}, batch, ["p", "q"])
    assert builder.call_args.args[1] is batch


class UnmodifiedAdapter:
    propose_new_texts = None

    def __init__(self, layout="reordered"):
        self.layout = layout
        self.parents = 0
        self.evaluations = []
        self.builder_batches = []

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluations.append((list(batch), candidate, capture_traces))
        if capture_traces and candidate["p"] == "seed":
            self.parents += 1
        score = 1 if candidate["p"] == "child" or self.parents == 1 else 0
        return EvaluationBatch(
            outputs=[None] * len(batch),
            scores=[score] * len(batch),
            trajectories=[{"task": task["label"]} for task in batch] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, batch, components):
        assert not hasattr(batch, "task_ids")
        self.builder_batches.append(batch)
        records = [{"Feedback": {"owner": t["task"], "step": step}} for t in batch.trajectories for step in range(2)]
        if self.layout == "filtered":
            records = [r for r in records if r["Feedback"]["owner"] != "task-A"]
        elif self.layout == "repeated":
            records = records + records
        elif self.layout == "empty":
            records = []
        elif self.layout == "error":
            raise RuntimeError("No valid predictions found for any module.")
        else:
            records.reverse()
        return dict.fromkeys(components, records)


@pytest.mark.parametrize("mode", ["curator", "disabled", "ordinary", "error"])
@pytest.mark.parametrize("layout", ["reordered", "filtered", "repeated"])
def test_core_uses_original_adapter_and_keeps_rollouts_and_llm_batched(tmp_path, mode, layout):
    adapter = UnmodifiedAdapter("error" if mode == "error" else layout)
    sampler = (
        EpochShuffledBatchSampler(minibatch_size=2)
        if mode == "ordinary"
        else TaskCuratorSampler(
            minibatch_size=2,
            lm=Mock(return_value='{"decision":"run","selected_success":"before"}'),
            use_reflection_anchors=mode != "disabled",
        )
    )
    seen = []
    reflection_lm = Mock(return_value="```\nchild\n```")

    def enrich(**kwargs):
        assert len(kwargs["eval_batch"].scores) == 2
        seen.append(deepcopy(kwargs["reflective_dataset"]))
        return kwargs["reflective_dataset"]

    def optimize():
        return gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=NamedLoader(),
            valset=[{"label": "val"}],
            adapter=adapter,
            batch_sampler=sampler,
            reflection_lm=reflection_lm,
            acceptance_criterion=SimpleNamespace(should_accept=lambda *_: False),
            reflective_dataset_enricher=enrich,
            run_dir=str(tmp_path),
            stop_callbacks=lambda st: len(st.full_program_trace) >= 2,
            display_progress_bar=False,
            raise_on_exception=False,
            skip_perfect_score=False,
        )

    if mode == "error":
        with pytest.raises(TaskCuratorContractError, match="single-task reflection"):
            optimize()
        reflection_lm.assert_not_called()
        return
    optimize()
    assert len(seen) == reflection_lm.call_count == 2
    assert len(adapter.builder_batches) == (4 if mode == "curator" else 2)
    assert all(len(b.scores) == (1 if mode == "curator" else 2) for b in adapter.builder_batches)
    # One seed validation, two parent minibatches and two child minibatches.
    assert len(adapter.evaluations) == 5
    assert all(len(batch) == 2 for batch, _, capture in adapter.evaluations if capture)
    for records in seen[1].values():
        for record in records:
            if mode == "curator":
                assert record["HistoricalSuccess"]["successful_state"]["task"] == record["Feedback"]["owner"]
            else:
                assert "HistoricalSuccess" not in record
            assert "_gepa_task_id" not in record


@pytest.mark.parametrize("kind", ["default", "confidence", "maths", "rag", "optimize_anything"])
def test_unmodified_builtin_adapter_anchor_alignment(anchored, kind):
    sampler, state, ids, _ = anchored
    components = ["p"]
    expected_ids = ids
    if kind in ("default", "confidence", "maths"):
        module, name = {
            "default": ("default_adapter", "DefaultAdapter"),
            "confidence": ("confidence_adapter", "ConfidenceAdapter"),
            "maths": ("anymaths_adapter", "AnyMathsAdapter"),
        }[kind]
        adapter = object.__new__(getattr(import_module(f"gepa.adapters.{module}.{module}"), name))
        trajectories = [
            {
                "data": {"input": tid, "answer": "42", "additional_context": {}},
                "full_assistant_response": "wrong",
                "parsed_value": "wrong",
                "logprob_score": None,
                "feedback": "failed",
            }
            for tid in ids
        ]
    elif kind == "rag":
        from gepa.adapters.generic_rag_adapter.generic_rag_adapter import GenericRAGAdapter

        adapter = object.__new__(GenericRAGAdapter)
        trajectories = [{"input": tid} for tid in ids]
        adapter._create_component_example = lambda comp, traj, *args: (
            {"Inputs": traj["input"]} if traj["input"] == ids[1] else None
        )
        expected_ids = [ids[1]]
        components = ["p", "q"]
    else:
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter

        adapter = object.__new__(OptimizeAnythingAdapter)
        trajectories = [{"Inputs": tid, "Feedback": "failed"} for tid in ids]
    batch = EvaluationBatch(scores=[0, 0], outputs=[None, None], trajectories=trajectories)
    result = build((sampler, state, ids, batch), adapter.make_reflective_dataset, components=components)
    assert set(result) == set(components)
    for records in result.values():
        assert [r["Inputs"] for r in records] == expected_ids
        for record in records:
            assert record["HistoricalSuccess"]["successful_state"]["task"] == record["Inputs"]


@pytest.mark.parametrize("shape", ["mapping", "component"])
def test_empty_task_skips_only_reflection_and_keeps_history(anchored, shape):
    sampler, state, ids, _ = anchored
    history = deepcopy(sampler.history)
    pool = dict(sampler._branch.retained_success)

    def builder(candidate, batch, components):
        tid = batch.trajectories[0]["task"]
        if tid == ids[0]:
            return {} if shape == "mapping" else {"p": [], "q": []}
        return {"p": [{"owner": tid}], "q": []}

    result = build(anchored, builder)
    assert set(result) == {"p"}
    assert result["p"][0]["owner"] == ids[1]
    assert result["p"][0]["HistoricalSuccess"]["successful_state"]["task"] == ids[1]
    assert sampler.history == history
    assert sampler._branch.retained_success == pool
    assert state.full_program_trace[-1]["task_curator_reflection_skips"] == [
        {"task_id": ids[0], "reason": "no_reflection_material"}
    ]
    assert [r["task_id"] for r in state.full_program_trace[-1]["task_curator_reflection_anchors"]] == [ids[1]]


def test_empty_raw_trace_can_still_supply_reflection_feedback(anchored):
    sampler, state, ids, batch = anchored
    batch = replace(batch, trajectories=[{"trace": [], "error": "action unavailable"}] * 2)
    result = build(
        (sampler, state, ids, batch),
        lambda _, single, __: {"p": [{"Feedback": single.trajectories[0]["error"]}]},
    )
    assert len(result["p"]) == 2
    assert all(row["Feedback"] == "action unavailable" for row in result["p"])
    assert state.full_program_trace[-1]["task_curator_reflection_skips"] == []


@pytest.mark.parametrize("proposal_owner", ["lm", "custom", "adapter"])
def test_whole_empty_batch_skips_mutation_and_preserves_evaluations(tmp_path, proposal_owner):
    adapter = UnmodifiedAdapter("empty")
    sampler = TaskCuratorSampler(minibatch_size=2, lm=Mock(return_value='{"decision":"run"}'))
    reflection = Mock(side_effect=AssertionError("Empty jobs must not reach reflection"))
    skipped, started, traces = [], [], []
    kwargs = {"reflection_lm": reflection}
    if proposal_owner == "custom":
        kwargs["custom_candidate_proposer"] = reflection
    elif proposal_owner == "adapter":
        adapter.propose_new_texts = reflection

    def stop(state):
        if len(state.full_program_trace) < 2:
            return False
        traces.extend(deepcopy(state.full_program_trace))
        return True

    result = gepa.optimize(
        seed_candidate={"p": "seed"},
        trainset=NamedLoader(),
        valset=[{"label": "val"}],
        adapter=adapter,
        batch_sampler=sampler,
        run_dir=str(tmp_path),
        stop_callbacks=stop,
        callbacks=[SimpleNamespace(on_evaluation_skipped=skipped.append, on_proposal_start=started.append)],
        display_progress_bar=False,
        raise_on_exception=True,
        skip_perfect_score=False,
        **kwargs,
    )
    reflection.assert_not_called()
    assert not started
    assert len(skipped) == 2
    assert all(event["reason"] == "no_reflection_material" for event in skipped)
    assert len(adapter.evaluations) == 3  # seed validation + two parent minibatches; no children
    assert result.total_metric_calls == 5  # the skipped tasks still count
    assert result.candidates == [{"p": "seed"}]
    assert all(len(sampler.history[tid]) >= 2 for tid in NamedLoader().all_ids())
    for entry in traces:
        assert len(entry["task_curator_results"]) == 2
        assert len(entry["task_curator_reflection_skips"]) == 2
        assert entry["task_curator_reflection_anchors"] == []
        assert "task_curator_mutation_results" not in entry


@pytest.fixture(params=["module", "program", "external"])
def dspy_reflection_builder(request):
    """Exercise the unchanged real builders; no API/LM calls are made."""
    dspy = pytest.importorskip("dspy")
    from random import Random

    modules = {
        "module": "gepa.adapters.dspy_adapter.dspy_adapter",
        "program": "gepa.adapters.dspy_full_program_adapter.full_program_adapter",
        "external": "dspy.teleprompt.gepa.gepa_utils",
    }
    adapter = object.__new__(import_module(modules[request.param]).DspyAdapter)
    predictor = dspy.Predict("question -> answer")
    other = dspy.Predict("question -> category")
    program = SimpleNamespace(named_predictors=lambda: [("p", predictor), ("q", other)])
    adapter.build_program = Mock(return_value=(program, None) if request.param == "program" else program)
    adapter.add_format_failure_as_feedback = False
    adapter.rng = Random(0)
    adapter.custom_instruction_proposer = None
    adapter.warn_on_score_mismatch = False
    adapter._flex_paths = {}
    adapter.feedback_map = {"p": lambda **_: {"score": 0, "feedback": "incorrect"}}

    def trajectory(tid, *, empty=False, unrelated=False):
        return {
            "trace": [] if empty else [(other if unrelated else predictor, {"question": tid}, {"answer": "wrong"})],
            "example": dspy.Example(question=tid).with_inputs("question"),
            "prediction": dspy.Prediction(answer="wrong"),
            "score": 0,
        }

    return adapter, trajectory, "program" if request.param == "program" else "p"


@pytest.mark.parametrize("empty_indices", [(0,), (1,), (0, 1)])
def test_dspy_no_material_skips_correct_task_and_anchor(anchored, dspy_reflection_builder, empty_indices):
    sampler, state, ids, _ = anchored
    adapter, trajectory, component = dspy_reflection_builder
    batch = EvaluationBatch(
        outputs=[None, None],
        scores=[0, 0],
        trajectories=[trajectory(tid, empty=i in empty_indices) for i, tid in enumerate(ids)],
    )
    history = deepcopy(sampler.history)
    result = build((sampler, state, ids, batch), adapter.make_reflective_dataset, components=[component])
    expected = [tid for i, tid in enumerate(ids) if i not in empty_indices]
    rows = result.get(component, [])
    field = "Program Inputs" if component == "program" else "Inputs"
    assert [r[field]["question"] for r in rows] == expected
    assert [r["HistoricalSuccess"]["successful_state"]["task"] for r in rows] == expected
    assert [r["task_id"] for r in state.full_program_trace[-1]["task_curator_reflection_skips"]] == [
        tid for i, tid in enumerate(ids) if i in empty_indices
    ]
    assert sampler.history == history


def test_dspy_trace_without_selected_component_is_empty_material(anchored, dspy_reflection_builder):
    sampler, state, ids, _ = anchored
    adapter, trajectory, component = dspy_reflection_builder
    batch = EvaluationBatch(
        outputs=[None, None],
        scores=[0, 0],
        trajectories=[trajectory(ids[0]), trajectory(ids[1], unrelated=True)],
    )
    result = build((sampler, state, ids, batch), adapter.make_reflective_dataset, components=[component])
    # Full-program reflection uses all modules; per-module reflection only uses p.
    expected = ids if component == "program" else ids[:1]
    assert [r["HistoricalSuccess"]["successful_state"]["task"] for r in result[component]] == expected


@pytest.mark.parametrize("failure", ["malformed", "unknown_component", "callback"])
def test_dspy_real_errors_are_not_mistaken_for_no_material(anchored, dspy_reflection_builder, failure):
    sampler, state, ids, _ = anchored
    adapter, trajectory, component = dspy_reflection_builder
    rows = [trajectory(tid) for tid in ids]
    error = Exception("No valid predictions found for any module.")
    if failure == "malformed":
        del rows[1]["prediction"]
    elif failure == "unknown_component":
        component = "missing"
    else:
        # Even the exact same generic exception from a callback is a real error.
        adapter.build_program = Mock(side_effect=error)
    batch = EvaluationBatch(outputs=[None, None], scores=[0, 0], trajectories=rows)
    with pytest.raises(TaskCuratorContractError) as caught:
        build((sampler, state, ids, batch), adapter.make_reflective_dataset, components=[component])
    if failure == "callback":
        assert caught.value.__cause__ is error
    else:
        assert isinstance(caught.value.__cause__, KeyError | AssertionError)


def test_component_without_material_is_not_sent_to_custom_proposer(tmp_path):
    adapter = UnmodifiedAdapter()

    def builder(candidate, batch, components):
        return {"p": [], "q": [{"Feedback": t["task"]} for t in batch.trajectories]}

    adapter.make_reflective_dataset = builder
    proposer = Mock(return_value={"q": "child"})
    gepa.optimize(
        seed_candidate={"p": "seed", "q": "seed"},
        trainset=NamedLoader(),
        valset=[{"label": "val"}],
        adapter=adapter,
        batch_sampler=TaskCuratorSampler(minibatch_size=2, lm=Mock(return_value='{"decision":"run"}')),
        module_selector=lambda *args: ["p", "q"],
        custom_candidate_proposer=proposer,
        acceptance_criterion=SimpleNamespace(should_accept=lambda *_: False),
        run_dir=str(tmp_path),
        stop_callbacks=lambda state: len(state.full_program_trace) >= 1,
        display_progress_bar=False,
        raise_on_exception=True,
        skip_perfect_score=False,
    )
    proposer.assert_called_once()
    assert proposer.call_args.args[2] == ["q"]
    assert set(proposer.call_args.args[1]) == {"q"}
    child = adapter.evaluations[-1][1]
    assert child == {"p": "seed", "q": "child"}
