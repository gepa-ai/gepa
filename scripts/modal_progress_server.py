"""
Modal web service serving the TurboGEPA dashboard + evolution JSON from the shared volume.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
EVOLUTION_PATH = Path("/mnt/turbogepa/evolution")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
    .add_local_file(REPO_ROOT / "scripts" / "evolution_live_v2.html", "/app/evolution_live_v2.html", copy=True)
)

app = modal.App("turbogepa-progress", image=image)
volume = modal.Volume.from_name("turbo-gepa-cache", create_if_missing=True)

fastapi_app = FastAPI()

def _load_run_file(run_id: str) -> dict:
    target = EVOLUTION_PATH / f"{run_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    try:
        return json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Malformed run file '{target.name}'") from exc

@fastapi_app.get("/")
async def serve_dashboard():
    return FileResponse("/app/evolution_live_v2.html")

@fastapi_app.get("/evolution")
async def evolution_api(run: str = "current"):
    try:
        volume.reload()
    except Exception:
        pass

    if not EVOLUTION_PATH.exists():
        raise HTTPException(status_code=404, detail="No evolution data available yet")

    if run == "current":
        current_path = EVOLUTION_PATH / "current.json"
        if not current_path.exists():
            raise HTTPException(status_code=404, detail="current.json not found")
        try:
            payload = json.loads(current_path.read_text())
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="Malformed current.json") from exc
        run = payload.get("run_id") or ""
        if not run:
            raise HTTPException(status_code=500, detail="current.json missing run_id")

    data = _load_run_file(run)
    return JSONResponse(data, headers={"Cache-Control": "no-store"})

@app.function(volumes={"/mnt/turbogepa": volume})
@modal.asgi_app()
def progress_service():  # pragma: no cover - exercised via Modal runtime
    return fastapi_app
