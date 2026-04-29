# ACM CAIS '26 Artifact Evaluation — `optany`

Artifact for the paper **"optany: Unified Text Optimization can Outperform Specialized Systems"** (ACM CAIS 2026).

## Review Modes

This artifact supports two review modes:

1. **Offline mode (recommended if you do not have API credits).**
   Inspect bundled trajectories, checkpoints, saved outputs, and post-hoc evaluations without making any model calls.
2. **Live rerun mode.**
   Re-run selected domains with your own API keys and hardware.

## What this artifact contains

- `domains/` — one self-contained subdirectory per experimental domain claimed in the paper. Each domain folder is independent and includes its own `README.md`, runnable code, and any artifacts needed to verify the paper's claims.
- `OFFLINE_ARTIFACTS.md` — zero-API guide to bundled trajectories, checkpoints, and saved evaluations.

## Quick repository map

| Paper section | Folder |
|---|---|
| §5.1 Coding Agent Skills | `domains/gskill/` |
| §5.2 Cloud Scheduling (CloudCast, Can't Be Late) | `domains/cloud_scheduling/` |
| §5.3 ARC-AGI Agent Architecture | `domains/arc_agi/` |
| §5.4 AIME Prompt Optimization | `domains/aime_math/` |
| §5.5 CUDA Kernel Generation (KernelBench) | `domains/kernelbench/` |
| §5.6 Circle Packing (n=26) | `domains/circle_packing/` |
| §5.7 SVG / CAD Image Generation | `domains/svg_cad/` |
| Appendix B Black-box Math Optimization | `domains/blackbox/` |
| Appendix C 3D Modeling (seedless) | `domains/modeling_3d/` |

## Offline-First Review Path

If you do not have API credits, start here:

1. `OFFLINE_ARTIFACTS.md`
2. `domains/aime_math/logs/`
3. `domains/arc_agi/logs/`
4. `domains/blackbox/logs/`
5. `domains/circle_packing/logs/`
6. `domains/cloud_scheduling/cloudcast/offline_logs/cloudcast_output.log`
7. `domains/gskill/offline_runs/`

These paths let reviewers inspect saved trajectories, checkpoints, best artifacts, and post-hoc evaluations without making model calls.

Best offline-supported domains:

- `domains/aime_math/`
- `domains/arc_agi/`
- `domains/blackbox/`
- `domains/circle_packing/`
- `domains/gskill/`
- `domains/cloud_scheduling/cloudcast/`

Still primarily live-rerun dependent or only partially supported offline:

- `domains/cloud_scheduling/can_be_late/` — code, traces, and plots are bundled, but not a full saved trajectory bundle.
- `domains/kernelbench/` — logs are bundled, but performance reproduction still requires a V100-class GPU.
- `domains/svg_cad/` and `domains/modeling_3d/` — useful tutorial assets are present, but they are not the strongest no-API validation path.

## Live Rerun Mode

For domains you want to re-run, first install dependencies:

```bash
uv sync --extra dev
```

Then follow the domain-specific README:

```bash
cd domains/<name>
export <PROVIDER>_API_KEY=<your-key>     # see the domain README
uv run python main.py
```

Where bundled state is provided, `main.py` is wired so `EngineConfig.run_dir` points at `./logs/`, and GEPA resumes from that checkpoint rather than starting from scratch. To run a fresh optimization, move or delete `logs/gepa_state.bin` first.

§5.5 (CUDA Kernel Generation, KernelBench) requires a V100 32GB GPU to re-run; see `domains/kernelbench/README.md`.

## Badges requested

Artifact Available, Functional, and Results Reproduced.
