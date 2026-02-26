"""Agentic RAG for KernelBench.

Searches local CUDA/PyTorch documentation and summarizes relevant snippets
to help generate correct, fast CUDA kernels. Uses an iterative
search-summarize-refine loop powered by a lightweight LM.

Public API:
    retrieve_cuda_docs(eval_result, ref_arch) — called by the evaluator to get
        targeted docs based on kernel evaluation outcome (error fix or optimization).
    agentic_retrieve(task) — lower-level entry point for arbitrary RAG queries.
"""

from pathlib import Path

import dspy

RAG_CONTENT_PATH = Path(__file__).parent / "rag_content"

CONTEXT = """You are the retrieval agent for KernelBench, a benchmark for generating fast CUDA kernels.
Your job: search docs and produce summaries that help generate correct, fast PyTorch CUDA extensions using load_inline.
Your summary will be injected into the kernel generator's prompt."""


def search_docs(term: str, max_results: int = 2, context: int = 3000) -> list[str]:
    """Search local rag_content files for *term* and return surrounding snippets.

    Walks all .md/.txt/.html files under ``rag_content/``, returning up to
    *max_results* snippets of *context* characters around the first match in each file.
    """
    results = []
    for f in RAG_CONTENT_PATH.rglob("*"):
        if f.is_file() and f.suffix in [".md", ".txt", ".html"]:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                pos = content.lower().find(term.lower())
                if pos >= 0:
                    start, end = max(0, pos - context), min(len(content), pos + context)
                    snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
                    results.append(f"[{f.name}]\n{snippet}")
                    if len(results) >= max_results:
                        break
            except Exception:
                continue
    return results


class SearchSig(dspy.Signature):
    __doc__ = CONTEXT + "\nGenerate search keywords for CUDA/PyTorch documentation."
    task: str = dspy.InputField(desc="Error message or kernel task")
    search_terms: str = dspy.OutputField(desc="3-5 SHORT keywords, comma-separated")


class SynthesizeSig(dspy.Signature):
    __doc__ = CONTEXT + "\nSummarize retrieved docs into actionable guidance."
    task: str = dspy.InputField(desc="Error message or kernel task")
    docs: str = dspy.InputField(desc="Retrieved documentation snippets")
    summary: str = dspy.OutputField(desc="Technical summary (max 3000 tokens): code patterns, headers, pitfalls")


class RefineSig(dspy.Signature):
    __doc__ = CONTEXT + "\nDecide if more docs are needed."
    task: str = dspy.InputField(desc="Error message or kernel task")
    summary: str = dspy.InputField(desc="Current summary")
    search_terms: str = dspy.OutputField(desc="2-3 NEW keywords, or 'DONE' if sufficient")


class AgenticRAG(dspy.Module):
    """Iterative search-summarize-refine loop over local docs.

    Each round: generate search keywords → search local files → synthesize a
    summary. After each round a refine step decides whether to search again
    (new keywords) or stop ("DONE"). Runs for at most *max_rounds* iterations.
    """

    def __init__(self, max_rounds: int = 3):
        super().__init__()
        self.max_rounds = max_rounds
        self.search = dspy.Predict(SearchSig)
        self.summarize = dspy.Predict(SynthesizeSig)
        self.refine = dspy.Predict(RefineSig)

    def forward(self, task: str, verbose: bool = False) -> str:
        summary = ""
        seen = set()

        for i in range(self.max_rounds):
            # Generate search terms
            if i == 0:
                terms = self.search(task=task).search_terms or ""
            else:
                terms = self.refine(task=task, summary=summary).search_terms or ""
                if "DONE" in terms.upper():
                    break

            if verbose:
                print(f"[Round {i+1}] Search: {terms}")

            # Execute search
            snippets = []
            for term in [t.strip() for t in terms.split(",") if t.strip()][:5]:
                for s in search_docs(term):
                    if s not in seen:
                        snippets.append(s)
                        seen.add(s)

            if verbose:
                print(f"[Round {i+1}] Found {len(snippets)} new snippets")

            if not snippets and i > 0:
                break

            # Synthesize
            if snippets:
                docs = (summary + "\n\n---NEW DOCS---\n\n" if summary else "") + "\n\n".join(snippets)
                summary = self.summarize(task=task, docs=docs[:12000]).summary or ""

        return summary or "No relevant documentation found."


def agentic_retrieve(task: str, verbose: bool = False, max_rounds: int = 1) -> str:
    """Run the agentic RAG loop for an arbitrary *task* description.

    Creates a lightweight LM context and returns a synthesized summary of
    relevant CUDA/PyTorch documentation. Defaults to a single search round.
    """
    lm = dspy.LM("openai/gpt-5-nano", temperature=1, max_tokens=16000)
    rag = AgenticRAG(max_rounds=max_rounds)
    with dspy.context(lm=lm):
        return rag(task=task, verbose=verbose)


def retrieve_cuda_docs(eval_result: dict, ref_arch: str) -> str:
    """Retrieve targeted CUDA docs based on kernel evaluation outcome.

    Called by the evaluator after each kernel execution. The query strategy
    depends on how the kernel failed or succeeded:
    - Error present → search for a fix using the error detail + reference arch.
    - Correct kernel → search for optimization patterns for the reference arch.
    - Other (e.g. shape mismatch without error detail) → return empty string.

    The returned docs are placed in ``side_info["cuda_docs"]`` so that both
    the reflection LM and the refiner can see them.
    """
    error_detail = eval_result.get("ErrorDetail", "")
    if error_detail:
        return agentic_retrieve(f"Fix error:\n{error_detail}\nTask:\n{ref_arch}")

    # Correct but not yet fast — look for optimization tips
    if eval_result.get("CorrectnessSucceeded", False):
        return agentic_retrieve(f"Optimize CUDA kernel for:\n{ref_arch}")

    return ""
