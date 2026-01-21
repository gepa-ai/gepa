"""Shared configuration for can't-be-late trace handling."""

# Sample trace IDs used for evaluation
TRACE_SAMPLE_IDS: list[int] = [
    0,
    8,
    9,
    20,
    21,
    33,
    42,
    51,
    61,
    70,
    99,
    107,
    117,
    126,
    135,
    145,
    154,
    163,
    172,
    182,
    191,
    219,
    228,
    238,
    247,
    256,
    266,
    275,
    284,
    294,
]

# Overheads that have been extracted in the reference archive.
TRACE_OVERHEADS: list[float] = [0.02, 0.20, 0.40]

# Environments used for test evaluation
LEGACY_ENV_PATHS = [
    "us-west-2a_k80_1",
    "us-west-2b_k80_1",
    "us-west-2a_v100_1",
    "us-west-2b_v100_1",
]

# Legacy evaluator caps to at most 30 traces per (env, overhead)
LEGACY_TRACE_TARGET = 30

