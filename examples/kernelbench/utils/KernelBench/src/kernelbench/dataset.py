################################################################################
# Unified Dataset Abstraction for KernelBench
#
# Supports both local filesystem and HuggingFace datasets through a unified
# interface. All problem access is by logical problem_id (1-indexed).
################################################################################

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional
import os
import random
import re
import hashlib

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


################################################################################
# Problem Dataclass
################################################################################

@dataclass
class Problem:
    """Unified representation of a KernelBench problem.
    
    Attributes:
        problem_id: 1-indexed logical ID (matches filename prefix)
        name: Filename, e.g., "1_Square_matrix_multiplication_.py"
        code: The actual source code
        level: KernelBench level (1, 2, 3, or custom)
        path: Local filesystem path (None if from HuggingFace)
        metadata: Extra metadata for future use (e.g., categories, difficulty)
    
    Note:
        Code is loaded eagerly when the dataset is constructed (~500KB for level 1).
        If memory becomes a concern for very large datasets, this could be refactored
        to lazy loading where code is only read when Problem.code is accessed.
    """
    problem_id: int
    name: str
    code: str
    level: int
    path: Optional[str] = None
    metadata: Optional[dict] = None

    @property
    def hash(self) -> str:
        """Compute code hash for problem identification.
        
        The hash ignores comments and whitespace, so functionally
        equivalent code produces the same hash. Useful for:
        - Deduplication across dataset versions
        - Tracking problem identity when code formatting changes
        - Comparing local vs HuggingFace versions
        """
        return get_code_hash(self.code)


################################################################################
# Hash Utilities
################################################################################

def get_code_hash(code: str) -> str:
    """Compute a unique hash for code, ignoring comments and whitespace."""
    # Remove multi-line comments
    code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", code)
    # Remove inline comments and all whitespace
    cleaned = re.sub(r"#.*$|\s+", "", code, flags=re.MULTILINE)
    return hashlib.md5(cleaned.encode()).hexdigest()


################################################################################
# Base Dataset Abstract Class
################################################################################

class BaseDataset(ABC):
    """Abstract base for all KernelBench datasets.
    
    Provides a unified interface for accessing problems by ID,
    iteration, and length.
    """

    @property
    @abstractmethod
    def level(self) -> int:
        """Return the KernelBench level."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Problem]:
        """Iterate over Problem objects in the dataset."""
        pass

    @abstractmethod
    def get_problem_by_id(self, problem_id: int) -> Problem:
        """Get problem by 1-indexed logical ID."""
        pass

    @abstractmethod
    def get_problem_ids(self) -> list[int]:
        """Get sorted list of all problem IDs in the dataset."""
        pass


    def subset(
        self,
        problem_ids: Optional[list[int]] = None,
        id_range: Optional[tuple[int, int]] = None,
    ) -> "BaseDataset":
        """Create a subset by problem IDs.
        
        Args:
            problem_ids: Specific problem IDs to include (e.g., [1, 3, 5])
            id_range: (start_id, end_id) inclusive range of problem IDs
            
        Returns:
            New dataset with only the specified problems
            
        Example:
            >>> dataset.subset(problem_ids=[1, 3, 5])
            >>> dataset.subset(id_range=(1, 10))
        """
        raise NotImplementedError("Subclasses should implement subset()")

    def sample(self, n: int, seed: int = 42) -> "BaseDataset":
        """Get a random sample of N problems.
        
        Args:
            n: Number of problems to sample
            seed: Random seed for reproducibility
            
        Returns:
            New dataset with N randomly selected problems
        """
        all_ids = self.get_problem_ids()
        n = min(n, len(all_ids))
        random.seed(seed)
        sampled_ids = random.sample(all_ids, n)
        return self.subset(problem_ids=sorted(sampled_ids))

    def get_representative_subset(self) -> "BaseDataset":
        """Get a curated representative subset for quick iteration.
        
        Returns a diverse subset covering different problem categories
        (matmul, conv, norms, etc.). Useful for testing.
        """
        rep_ids = {
            1: [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87],
            2: [1, 2, 8, 18, 23, 28, 33, 43],
            3: [1, 5, 8, 11, 20, 21, 33, 38, 43],
        }
        
        if self.level not in rep_ids:
            raise ValueError(f"No representative subset for level {self.level}")
        
        available_ids = set(self.get_problem_ids())
        subset_ids = [pid for pid in rep_ids[self.level] if pid in available_ids]
        
        return self.subset(problem_ids=subset_ids)


################################################################################
# Local Filesystem Dataset
################################################################################

class LocalKernelBenchDataset(BaseDataset):
    """Dataset backed by local filesystem.
    
    Loads problems from KernelBench/level{N}/*.py
    Flexible for any level number (1, 2, 3, or custom levels).
    """

    def __init__(
        self,
        level: int,
        base_path: str = KERNEL_BENCH_PATH,
        problem_ids: Optional[list[int]] = None,
        id_range: Optional[tuple[int, int]] = None,
    ):
        """Initialize local dataset.
        
        Args:
            level: KernelBench level (any positive integer)
            base_path: Path to KernelBench directory
            problem_ids: Optional list of specific problem IDs to include
            id_range: Optional (start_id, end_id) inclusive range
        """
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")

        self._level = level
        self._base_path = base_path
        self._problems: dict[int, Problem] = {}
        
        # Build filter set from problem_ids and/or id_range
        self._filter_ids = self._build_filter_set(problem_ids, id_range)
        self._load_problems()

    def _build_filter_set(
        self,
        problem_ids: Optional[list[int]],
        id_range: Optional[tuple[int, int]],
    ) -> Optional[set[int]]:
        """Build a set of IDs to filter by, or None for no filtering."""
        if problem_ids is None and id_range is None:
            return None
        
        filter_set = set()
        if problem_ids:
            filter_set.update(problem_ids)
        if id_range:
            start, end = id_range
            filter_set.update(range(start, end + 1))
        return filter_set

    @property
    def level(self) -> int:
        return self._level

    def _load_problems(self):
        problem_dir = os.path.join(self._base_path, f"level{self._level}")
        
        if not os.path.exists(problem_dir):
            raise FileNotFoundError(f"Problem directory not found: {problem_dir}")

        for file_name in os.listdir(problem_dir):
            if not file_name.endswith(".py"):
                continue

            try:
                problem_id = int(file_name.split("_")[0])
            except (ValueError, IndexError):
                continue

            # Apply filter if specified
            if self._filter_ids is not None and problem_id not in self._filter_ids:
                continue

            path = os.path.join(problem_dir, file_name)
            with open(path, "r") as f:
                code = f.read()

            self._problems[problem_id] = Problem(
                problem_id=problem_id,
                name=file_name,
                code=code,
                level=self._level,
                path=path,
            )

    def get_problem_by_id(self, problem_id: int) -> Problem:
        if problem_id not in self._problems:
            raise ValueError(f"Problem ID {problem_id} not found in dataset")
        return self._problems[problem_id]

    def get_problem_ids(self) -> list[int]:
        return sorted(self._problems.keys())

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self) -> Iterator[Problem]:
        for pid in self.get_problem_ids():
            yield self._problems[pid]

    def __repr__(self) -> str:
        return f"LocalKernelBenchDataset(level={self._level}, problems={len(self)})"

    def subset(
        self,
        problem_ids: Optional[list[int]] = None,
        id_range: Optional[tuple[int, int]] = None,
    ) -> "LocalKernelBenchDataset":
        """Create a subset of this dataset."""
        return LocalKernelBenchDataset(
            level=self._level,
            base_path=self._base_path,
            problem_ids=problem_ids,
            id_range=id_range,
        )


################################################################################
# HuggingFace Dataset
################################################################################

class HuggingFaceKernelBenchDataset(BaseDataset):
    """Dataset backed by HuggingFace datasets."""

    def __init__(
        self,
        level: int,
        dataset_name: str = "ScalingIntelligence/KernelBench",
        problem_ids: Optional[list[int]] = None,
        id_range: Optional[tuple[int, int]] = None,
    ):
        """Initialize HuggingFace dataset.
        
        Args:
            level: KernelBench level (1, 2, or 3)
            dataset_name: HuggingFace dataset identifier
            problem_ids: Optional list of specific problem IDs to include
            id_range: Optional (start_id, end_id) inclusive range
        """
        if level not in [1, 2, 3]:
            raise ValueError(f"HuggingFace dataset only has levels 1, 2, 3, got {level}")

        self._level = level
        self._dataset_name = dataset_name
        self._problems: dict[int, Problem] = {}
        self._filter_ids = self._build_filter_set(problem_ids, id_range)
        self._load_dataset()

    def _build_filter_set(
        self,
        problem_ids: Optional[list[int]],
        id_range: Optional[tuple[int, int]],
    ) -> Optional[set[int]]:
        """Build a set of IDs to filter by, or None for no filtering."""
        if problem_ids is None and id_range is None:
            return None
        
        filter_set = set()
        if problem_ids:
            filter_set.update(problem_ids)
        if id_range:
            start, end = id_range
            filter_set.update(range(start, end + 1))
        return filter_set

    @property
    def level(self) -> int:
        return self._level

    def _load_dataset(self):
        from datasets import load_dataset

        split_name = f"level_{self._level}"
        hf_dataset = load_dataset(self._dataset_name, split=split_name)

        for row in hf_dataset:
            problem_id = row["problem_id"]

            if self._filter_ids is not None and problem_id not in self._filter_ids:
                continue

            self._problems[problem_id] = Problem(
                problem_id=problem_id,
                name=row["name"],
                code=row["code"],
                level=self._level,
                path=None,
            )

    def get_problem_by_id(self, problem_id: int) -> Problem:
        if problem_id not in self._problems:
            raise ValueError(f"Problem ID {problem_id} not found in dataset")
        return self._problems[problem_id]

    def get_problem_ids(self) -> list[int]:
        return sorted(self._problems.keys())

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self) -> Iterator[Problem]:
        for pid in self.get_problem_ids():
            yield self._problems[pid]

    def __repr__(self) -> str:
        return f"HuggingFaceKernelBenchDataset(level={self._level}, problems={len(self)})"

    def subset(
        self,
        problem_ids: Optional[list[int]] = None,
        id_range: Optional[tuple[int, int]] = None,
    ) -> "HuggingFaceKernelBenchDataset":
        """Create a subset of this dataset."""
        return HuggingFaceKernelBenchDataset(
            level=self._level,
            dataset_name=self._dataset_name,
            problem_ids=problem_ids,
            id_range=id_range,
        )


################################################################################
# Factory Function
################################################################################

def construct_kernelbench_dataset(
    level: int,
    source: str = "local",
    dataset_name: str = "ScalingIntelligence/KernelBench",
    base_path: str = KERNEL_BENCH_PATH,
    problem_ids: Optional[list[int]] = None,
    id_range: Optional[tuple[int, int]] = None,
) -> BaseDataset:
    """Construct a KernelBench dataset for a specific level.
    
    Args:
        level: KernelBench level (1, 2, 3, or custom for local)
        source: "local" for filesystem, "huggingface" for HF datasets
        dataset_name: HuggingFace dataset identifier (if source="huggingface")
        base_path: Path to KernelBench directory (if source="local")
        problem_ids: Optional list of specific problem IDs to include
        id_range: Optional (start_id, end_id) inclusive range
        
    Returns:
        BaseDataset instance for the specified level
        
    Examples:
        # Local filesystem (default)
        >>> dataset = construct_kernelbench_dataset(level=1, source="local")
        >>> len(dataset)
        100
        
        # HuggingFace
        >>> dataset = construct_kernelbench_dataset(level=1, source="huggingface")
        >>> len(dataset)
        100
        
        # Filter by specific IDs
        >>> dataset = construct_kernelbench_dataset(level=1, problem_ids=[1, 3, 5])
        >>> dataset.get_problem_ids()
        [1, 3, 5]
        
        # Filter by range
        >>> dataset = construct_kernelbench_dataset(level=1, id_range=(1, 10))
        >>> dataset.get_problem_ids()
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Access problems
        >>> problem = dataset.get_problem_by_id(1)
        >>> problem.name
        '1_Square_matrix_multiplication_.py'
        >>> problem.code[:50]
        'import torch...'
    """
    if source == "local":
        return LocalKernelBenchDataset(level, base_path, problem_ids, id_range)
    elif source == "huggingface":
        return HuggingFaceKernelBenchDataset(level, dataset_name, problem_ids, id_range)
    else:
        raise ValueError(f"Unknown source: {source}. Must be 'local' or 'huggingface'")


################################################################################
# Convenience Functions
################################################################################

def fetch_ref_arch_from_dataset(
    dataset: BaseDataset,
    problem_id: int,
) -> tuple[Optional[str], str, str]:
    """Fetch reference architecture from dataset.

    Returns:
        (path, name, code) - path is None for HuggingFace datasets
    """
    problem = dataset.get_problem_by_id(problem_id)
    return (problem.path, problem.name, problem.code)


def get_kernelbench_subset(
    level: int,
    num_subset_problems: int = 10,
    random_seed: int = 42,
    source: str = "local",
) -> tuple[BaseDataset, list[int]]:
    """Get a random subset of problems.
    
    Returns:
        (subset_dataset, subset_problem_ids)
    """
    full_dataset = construct_kernelbench_dataset(level, source=source)
    all_ids = full_dataset.get_problem_ids()

    random.seed(random_seed)
    num_subset_problems = min(num_subset_problems, len(all_ids))
    subset_ids = sorted(random.sample(all_ids, num_subset_problems))

    subset_dataset = construct_kernelbench_dataset(
        level=level,
        source=source,
        problem_ids=subset_ids,
    )
    return subset_dataset, subset_ids


################################################################################
# Representative Subsets of KernelBench
# Use these for quick iteration without running the full dataset
################################################################################

# Level 1: Basic operators - matmul, activations, norms, pooling, convolutions
LEVEL1_REPRESENTATIVE_SUBSET = [
    "1_Square_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "6_Matmul_with_large_K_dimension_.py",
    "18_Matmul_with_transposed_both.py",
    "23_Softmax.py",
    "26_GELU_.py",
    "33_BatchNorm.py",
    "36_RMSNorm_.py",
    "40_LayerNorm.py",
    "42_Max_Pooling_2D.py",
    "48_Mean_reduction_over_a_dimension.py",
    "54_conv_standard_3D__square_input__square_kernel.py",
    "57_conv_transposed_2D__square_input__square_kernel.py",
    "65_conv_transposed_2D__square_input__asymmetric_kernel.py",
    "77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py",
    "82_conv_depthwise_2D_square_input_square_kernel.py",
    "86_conv_depthwise_separable_2D.py",
    "87_conv_pointwise_2D.py",
]
LEVEL1_REPRESENTATIVE_IDS = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]

# Level 2: Fused operators - multi-op fusion patterns
LEVEL2_REPRESENTATIVE_SUBSET = [
    "1_Conv2D_ReLU_BiasAdd.py",
    "2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
    "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py",
    "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
    "23_Conv3d_GroupNorm_Mean.py",
    "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
    "33_Gemm_Scale_BatchNorm.py",
    "43_Conv3d_Max_LogSumExp_ReLU.py",
]
LEVEL2_REPRESENTATIVE_IDS = [1, 2, 8, 18, 23, 28, 33, 43]

# Level 3: Full models - MLP, CNN architectures, RNNs, Transformers
LEVEL3_REPRESENTATIVE_SUBSET = [
    "1_MLP.py",
    "5_AlexNet.py",
    "8_ResNetBasicBlock.py",
    "11_VGG16.py",
    "20_MobileNetV2.py",
    "21_EfficientNetMBConv.py",
    "33_VanillaRNN.py",
    "38_LTSMBidirectional.py",
    "43_MinGPTCausalAttention.py",
]
LEVEL3_REPRESENTATIVE_IDS = [1, 5, 8, 11, 20, 21, 33, 38, 43]


def get_representative_dataset(level: int, source: str = "local") -> BaseDataset:
    """Get a representative subset dataset for quick iteration.
    
    Args:
        level: 1, 2, or 3
        source: "local" or "huggingface"
        
    Returns:
        Dataset containing only representative problems
    """
    id_map = {
        1: LEVEL1_REPRESENTATIVE_IDS,
        2: LEVEL2_REPRESENTATIVE_IDS,
        3: LEVEL3_REPRESENTATIVE_IDS,
    }
    if level not in id_map:
        raise ValueError(f"No representative subset for level {level}")
    
    return construct_kernelbench_dataset(
        level=level,
        source=source,
        problem_ids=id_map[level],
    )