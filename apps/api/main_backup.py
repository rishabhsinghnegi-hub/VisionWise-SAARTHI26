"""Local VisionWise API: webcam frame inference and reviewable event history."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sqlite3

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field

from inference import Detector
from inference import Detection
from rules import interpret, route_visual_question

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = Path(os.getenv("VISIONWISE_DB_PATH", str(BASE_DIR / "visionwise.db")))
MODEL_PATH = Path(os.getenv("VISIONWISE_MODEL_PATH", str(BASE_DIR.parent.parent / "yolo11n.pt")))
MAX_FRAME_BYTES = 8 * 1024 * 1024
logger = logging.getLogger("visionwise")
detector = Detector(MODEL_PATH)


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                mode TEXT NOT NULL,
                summary TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                inference_ms REAL NOT NULL,
                model_name TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_detection_events_created_at
                ON detection_events(created_at DESC);
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    init_db()
    detector.load()
    yield


app = FastAPI(title="VisionWise API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)


class Settings(BaseModel):
    confidence: float = Field(default=0.2, ge=0.05, le=0.95)
    mode: str = Field(default="everyday", pattern="^(everyday|navigation|indoor)$")
    save_history: bool = True
    wake_phrase: str = Field(default="Vision Wise", min_length=2, max_length=40)
    continuous_narration: bool = False
    speech_rate: float = Field(default=1.0, ge=0.7, le=1.3)
    speech_volume: float = Field(default=0.85, ge=0, le=1)
    speech_voice: str = Field(default="", max_length=160)


class NarrationRequest(BaseModel):
    detections: list[Detection] = Field(max_length=100)
    mode: str = Field(default="everyday", pattern="^(everyday|navigation|indoor)$")


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    detections: list[Detection] = Field(default_factory=list, max_length=100)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)
    vision_available: bool = False


@app.get("/")
def home():
    return {"name": "VisionWise API", "docs": "/docs", "scope": "Laptop webcam and local object detection"}


@app.get("/health")
async def health():
    model = detector.status()
    assistant_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    assistant_status = {"provider": "ollama", "model": assistant_model, "local": True, "available": False, "installed": False}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.0, connect=0.5)) as client:
            response = await client.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/tags")
        response.raise_for_status()
        tags = response.json().get("models", [])
        assistant_status["available"] = True
        assistant_status["installed"] = any(item.get("name") == assistant_model for item in tags)
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    return {
        "ok": True,
        "service": "available",
        "model": model,
        "assistant": assistant_status,
        "database": "available" if DB_PATH.exists() else "not_initialized",
        "hardware": {"camera": "browser webcam", "distance_sensor": "not_configured", "microcontroller": "out_of_scope"},
    }


@app.get("/settings")
def get_settings():
    defaults = Settings().model_dump()
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    for row in rows:
        if row["key"] in defaults:
            import json
            defaults[row["key"]] = json.loads(row["value"])
    return Settings(**defaults)


@app.put("/settings")
def put_settings(settings: Settings):
    import json
    with connect() as conn:
        for key, value in settings.model_dump().items():
            conn.execute(
                "INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                (key, json.dumps(value)),
            )
    return settings


@app.post("/api/detect")
async def detect_frame(
    request: Request,
    confidence: float = Query(default=0.2, ge=0.05, le=0.95),
    mode: str = Query(default="everyday", pattern="^(everyday|navigation|indoor)$"),
    save_history: bool = Query(default=True),
):
    content_type = request.headers.get("content-type", "")
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Send a JPEG, PNG, or WebP image frame.")
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image frame is empty.")
    if len(image_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="The image frame exceeds the 8 MB limit.")
    if not detector.status()["available"]:
        raise HTTPException(status_code=503, detail="The local YOLO model is unavailable. Check API setup and model path.")
    try:
        result = detector.predict(image_bytes, confidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Frame inference failed")
        raise HTTPException(status_code=500, detail="Frame inference failed. Check the API logs.") from exc
    detections = [item.model_dump() for item in result["detections"]]
    guidance = interpret(detections, mode)
    logger.info(
        "Frame analyzed: detections=%d relevant=%d mode=%s inference_ms=%.1f",
        len(detections), guidance["relevant_count"], mode, result["inference_ms"],
    )
    response = {
        "detections": detections,
        "guidance": guidance,
        "mode": mode,
        "inference_ms": result["inference_ms"],
        "model": result["model"],
        "event_id": None,
    }
    if save_history and guidance["relevant_count"] > 0:
        import json
        try:
            with connect() as conn:
                recent = conn.execute(
                    "SELECT created_at, summary FROM detection_events ORDER BY id DESC LIMIT 1"
                ).fetchone()
                # Suppress repeat writes for identical summaries during rapid camera polling.
                duplicate = recent and recent["summary"] == guidance["summary"] and conn.execute(
                    "SELECT strftime('%s','now') - strftime('%s',?) < 15", (recent["created_at"],)
                ).fetchone()[0]
                if not duplicate:
                    cursor = conn.execute(
                        "INSERT INTO detection_events(mode,summary,detections_json,inference_ms,model_name) VALUES(?,?,?,?,?)",
                        (mode, guidance["summary"], json.dumps(detections), result["inference_ms"], result["model"]["name"]),
                    )
                    response["event_id"] = cursor.lastrowid
        except sqlite3.Error:
            logger.exception("Could not persist detection event")
    return response


@app.post("/api/narrate")
async def narrate_detections(payload: NarrationRequest):
    detections = [item.model_dump() for item in payload.detections]
    guidance = interpret(detections, payload.mode)
    return {"message": guidance["summary"], "source": "local", "configured": False}


@app.post("/api/assistant")
async def ask_assistant(payload: AssistantRequest):
    """Chat with a local Ollama model; no API key or cloud service is used."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    detections = [item.model_dump() for item in payload.detections] if payload.vision_available else []
    routed_reply = route_visual_question(payload.message, detections, payload.vision_available)
    if routed_reply is not None:
        return {"reply": routed_reply, "model": "local-rules", "source": "local"}
    objects = ", ".join(
        f"{item.label} ({item.position}, confidence {item.confidence:.0%})"
        for item in detections[:30]
    ) or "No current camera snapshot is available."
    messages = [{
        "role": "system",
        "content": (
            "You are VisionWise, a friendly conversational assistant for a blind or low-vision user. "
            "Speak naturally, like a helpful person, and answer each question directly. Answer general knowledge questions "
            "as well as questions about the camera; use recent chat history to understand follow-up questions. Give a clear, "
            "speech-friendly answer in a few natural sentences, without canned greetings or unnecessary disclaimers. "
            "Current camera detections: " + objects + ". "
            "Ground visual claims only in the current detection list. Do not claim to see the camera image itself, "
            "invent objects, infer physical distance or movement, identify a physically nearest person, or suggest a safe route. "
            "Left and right refer to the image. Do not treat earlier chat messages as current camera observations. "
            "If current detections are unavailable, say that clearly."
        ),
    }]
    for item in payload.history[-10:]:
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:1000]})
    messages.append({"role": "user", "content": payload.message})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=2.0)) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
        return {"reply": answer, "model": model, "source": "ollama"}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama could not load {model}. Pull the model with `ollama pull {model}`.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Ollama is not reachable. Install Ollama, run it, and pull the selected model.") from exc


@app.get("/api/history")
def history(limit: int = Query(default=50, ge=1, le=200), q: str = Query(default="", max_length=80)):
    import json
    with connect() as conn:
        rows = conn.execute(
            "SELECT id,created_at,mode,summary,detections_json,inference_ms,model_name FROM detection_events "
            "WHERE summary LIKE ? ORDER BY id DESC LIMIT ?", (f"%{q}%", limit)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        avg = conn.execute("SELECT AVG(inference_ms) FROM detection_events").fetchone()[0]
    return {
        "items": [{**dict(row), "detections": json.loads(row["detections_json"])} for row in rows],
        "total": total,
        "average_inference_ms": round(avg, 1) if avg is not None else None,
    }


@app.delete("/api/history")
def clear_history():
    with connect() as conn:
        conn.execute("DELETE FROM detection_events")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='detection_events'")
    return {"cleared": True}


@app.get("/api/stats")
def stats():
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        today = conn.execute("SELECT COUNT(*) FROM detection_events WHERE date(created_at)=date('now')").fetchone()[0]
        average = conn.execute("SELECT AVG(inference_ms) FROM detection_events").fetchone()[0]
    return {"events": total, "events_today": today, "average_inference_ms": round(average, 1) if average is not None else None}
