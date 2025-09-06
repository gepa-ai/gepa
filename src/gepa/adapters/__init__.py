# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from . import default_adapter

try:
    from . import generic_rag_adapter
except ImportError:
    # RAG adapter dependencies not available
    pass