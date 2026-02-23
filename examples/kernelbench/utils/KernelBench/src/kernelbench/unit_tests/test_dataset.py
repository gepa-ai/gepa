"""
Unit tests for the KernelBench dataset module.

Usage:
    pytest src/kernelbench/unit_tests/test_dataset.py -v
"""

import pytest
from kernelbench.dataset import (
    get_code_hash,
    construct_kernelbench_dataset,
    fetch_ref_arch_from_dataset,
    Problem,
    BaseDataset,
    LocalKernelBenchDataset,
    LEVEL1_REPRESENTATIVE_IDS,
)


################################################################################
# Hash Tests
################################################################################

def test_get_code_hash_ignores_comments():
    """Hash should be equal for semantically equivalent code with different comments."""
    code_v1 = """
    import torch 
    # This is for a single batch
    '''
    Some random multi-line comment
    '''
    B = 1
    """
    
    code_v2 = """
    import torch 
    '''
    More problem descriptions (updated)
    '''
    # low batch setting

    B = 1
    """
    
    assert get_code_hash(code_v1) == get_code_hash(code_v2)


def test_get_code_hash_different_for_different_code():
    """Hash should differ for code with actual differences."""
    code_batch_1 = """
    import torch 
    B = 1
    """
    
    code_batch_64 = """
    import torch 
    B = 64
    """
    
    assert get_code_hash(code_batch_1) != get_code_hash(code_batch_64)


################################################################################
# Dataset Construction Tests
################################################################################

def test_construct_local_dataset():
    """Test constructing a local dataset."""
    dataset = construct_kernelbench_dataset(level=1, source="local")
    
    assert isinstance(dataset, BaseDataset)
    assert isinstance(dataset, LocalKernelBenchDataset)
    assert dataset.level == 1
    assert len(dataset) > 0


def test_construct_dataset_invalid_level():
    """Test that invalid level raises ValueError."""
    with pytest.raises(ValueError):
        construct_kernelbench_dataset(level=0, source="local")
    
    with pytest.raises(ValueError):
        construct_kernelbench_dataset(level=-1, source="local")


def test_construct_dataset_invalid_source():
    """Test that invalid source raises ValueError."""
    with pytest.raises(ValueError):
        construct_kernelbench_dataset(level=1, source="invalid")


################################################################################
# Problem Access Tests
################################################################################

def test_get_problem_by_id():
    """Test getting a problem by its logical ID."""
    dataset = construct_kernelbench_dataset(level=1)
    
    problem = dataset.get_problem_by_id(1)
    
    assert isinstance(problem, Problem)
    assert problem.problem_id == 1
    assert problem.name.startswith("1_")
    assert problem.level == 1
    assert len(problem.code) > 0
    assert problem.path is not None


def test_get_problem_by_id_not_found():
    """Test that non-existent ID raises ValueError."""
    dataset = construct_kernelbench_dataset(level=1)
    
    with pytest.raises(ValueError, match="not found"):
        dataset.get_problem_by_id(9999)


def test_get_problem_ids():
    """Test getting list of all problem IDs."""
    dataset = construct_kernelbench_dataset(level=1)
    
    ids = dataset.get_problem_ids()
    
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert ids == sorted(ids)  # should be sorted
    assert 1 in ids


def test_problem_hash():
    """Test that Problem.hash property works."""
    dataset = construct_kernelbench_dataset(level=1)
    problem = dataset.get_problem_by_id(1)
    
    hash_value = problem.hash
    
    assert isinstance(hash_value, str)
    assert len(hash_value) == 32  # MD5 hex digest length


################################################################################
# Subset Tests
################################################################################

def test_subset_by_ids():
    """Test creating a subset by specific problem IDs."""
    dataset = construct_kernelbench_dataset(level=1)
    subset = dataset.subset(problem_ids=[1, 3, 5])
    
    assert len(subset) == 3
    assert subset.get_problem_ids() == [1, 3, 5]


def test_subset_by_range():
    """Test creating a subset by ID range."""
    dataset = construct_kernelbench_dataset(level=1)
    subset = dataset.subset(id_range=(1, 5))
    
    assert len(subset) == 5
    assert subset.get_problem_ids() == [1, 2, 3, 4, 5]


def test_sample_random():
    """Test random sampling from dataset."""
    dataset = construct_kernelbench_dataset(level=1)
    
    sample1 = dataset.sample(n=5, seed=42)
    sample2 = dataset.sample(n=5, seed=42)
    sample3 = dataset.sample(n=5, seed=123)
    
    assert len(sample1) == 5
    assert sample1.get_problem_ids() == sample2.get_problem_ids()  # same seed
    assert sample1.get_problem_ids() != sample3.get_problem_ids()  # different seed


def test_get_representative_subset():
    """Test getting the representative subset."""
    dataset = construct_kernelbench_dataset(level=1)
    rep = dataset.get_representative_subset()
    
    assert len(rep) == len(LEVEL1_REPRESENTATIVE_IDS)
    assert rep.get_problem_ids() == LEVEL1_REPRESENTATIVE_IDS


################################################################################
# Iterator Tests
################################################################################

def test_dataset_iteration():
    """Test iterating over dataset."""
    dataset = construct_kernelbench_dataset(level=1, problem_ids=[1, 2, 3])
    
    problems = list(dataset)
    
    assert len(problems) == 3
    assert all(isinstance(p, Problem) for p in problems)
    assert [p.problem_id for p in problems] == [1, 2, 3]



def test_dataset_len():
    """Test len() on dataset."""
    dataset = construct_kernelbench_dataset(level=1, problem_ids=[1, 2, 3])
    assert len(dataset) == 3


################################################################################
# Compatibility Tests
################################################################################

def test_fetch_ref_arch_from_dataset():
    """Test the backward-compatible fetch function."""
    dataset = construct_kernelbench_dataset(level=1)
    
    path, name, code = fetch_ref_arch_from_dataset(dataset, problem_id=1)
    
    assert path is not None
    assert name.startswith("1_")
    assert len(code) > 0
    
    # Should match direct problem access
    problem = dataset.get_problem_by_id(1)
    assert path == problem.path
    assert name == problem.name
    assert code == problem.code


################################################################################
# Multiple Levels Tests
################################################################################

def test_all_levels_load():
    """Test that all standard levels can be loaded."""
    for level in [1, 2, 3]:
        dataset = construct_kernelbench_dataset(level=level)
        assert len(dataset) > 0
        assert dataset.level == level


################################################################################
# HuggingFace Tests (requires network)
################################################################################

@pytest.mark.slow
def test_huggingface_dataset_loads():
    """Test that HuggingFace dataset can be loaded."""
    dataset = construct_kernelbench_dataset(level=1, source="huggingface")
    
    assert len(dataset) > 0
    assert dataset.level == 1
    
    problem = dataset.get_problem_by_id(1)
    assert problem.problem_id == 1
    assert problem.name.startswith("1_")
    assert len(problem.code) > 0
    assert problem.path is None  # HF has no local path


@pytest.mark.slow
def test_local_and_huggingface_parity():
    """Test that local and HuggingFace datasets have the same content."""
    local_ds = construct_kernelbench_dataset(level=1, source="local")
    hf_ds = construct_kernelbench_dataset(level=1, source="huggingface")
    
    # Same number of problems
    assert len(local_ds) == len(hf_ds), "Local and HF should have same number of problems"
    
    # Same problem IDs
    assert local_ds.get_problem_ids() == hf_ds.get_problem_ids(), "Problem IDs should match"
    
    # Check a few problems have same content
    for pid in [1, 10, 50]:
        local_p = local_ds.get_problem_by_id(pid)
        hf_p = hf_ds.get_problem_by_id(pid)
        
        # Names may differ slightly (HF may not have .py extension)
        # But the base name (without extension) should match
        local_base = local_p.name.replace(".py", "")
        hf_base = hf_p.name.replace(".py", "")
        assert local_base == hf_base, f"Problem {pid} name mismatch: {local_p.name} vs {hf_p.name}"
        
        # Code and hash should match exactly
        assert local_p.code == hf_p.code, f"Problem {pid} code mismatch"
        assert local_p.hash == hf_p.hash, f"Problem {pid} hash mismatch"


@pytest.mark.slow
def test_huggingface_subset():
    """Test that HuggingFace dataset supports subsetting."""
    hf_ds = construct_kernelbench_dataset(level=1, source="huggingface")
    
    subset = hf_ds.subset(problem_ids=[1, 2, 3])
    assert len(subset) == 3
    assert subset.get_problem_ids() == [1, 2, 3]


@pytest.mark.slow
def test_huggingface_representative():
    """Test that HuggingFace dataset supports representative subset."""
    hf_ds = construct_kernelbench_dataset(level=1, source="huggingface")
    rep = hf_ds.get_representative_subset()
    
    assert len(rep) == len(LEVEL1_REPRESENTATIVE_IDS)


@pytest.mark.slow  
def test_unified_interface_behavior():
    """Test that both sources behave identically through the unified interface."""
    for source in ["local", "huggingface"]:
        dataset = construct_kernelbench_dataset(level=1, source=source)
        
        # All these operations should work the same
        assert len(dataset) > 0
        assert 1 in dataset.get_problem_ids()
        
        problem = dataset.get_problem_by_id(1)
        assert isinstance(problem, Problem)
        assert problem.problem_id == 1
        assert problem.level == 1
        
        # Iteration works
        count = 0
        for p in dataset:
            count += 1
            if count >= 3:
                break
        assert count == 3
        
        # Subset works
        subset = dataset.subset(problem_ids=[1, 2])
        assert len(subset) == 2