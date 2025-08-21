import os
import uuid
import time
import shutil
import asyncio
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field



Status = Literal["PENDING", "IN_PROGRESS", "COMPLETED"]

class CreateTaskIn(BaseModel):
    file_path: str = Field(..., description="Абсолютный путь к аудио файлу")

class Phrase(BaseModel):
    start: float
    end: float
    text: str

class TaskResult(BaseModel):
    phrases: List[Phrase]

class TaskState(BaseModel):
    task_id: str
    status: Status
    results: Optional[TaskResult] = None


app = FastAPI(title="Whisper Transcription API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QUEUE: List[str] = []
QUEUE_LOCK = asyncio.Lock()
TASKS: Dict[str, Dict[str, Any]] = {}

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
MODEL_IMPL: Optional[str] = None
WHISPER = None

SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm"}



def _load_model():

    global MODEL_IMPL, WHISPER
    try:
        from faster_whisper import WhisperModel
        device = "auto"
        WHISPER = WhisperModel(WHISPER_MODEL_NAME, device=device)
        MODEL_IMPL = "faster_whisper"
        return
    except Exception:
        pass

    try:
        import whisper
        WHISPER = whisper.load_model(WHISPER_MODEL_NAME)
        MODEL_IMPL = "openai_whisper"
        return
    except Exception:
        MODEL_IMPL = None
        WHISPER = None


@app.on_event("startup")
async def _startup():
    _load_model()
    asyncio.create_task(_worker_loop())



def _ext_ok(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in SUPPORTED_EXT

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

async def _queue_push(task_id: str):
    async with QUEUE_LOCK:
        QUEUE.append(task_id)

async def _queue_pop_left() -> Optional[str]:
    async with QUEUE_LOCK:
        if QUEUE:
            return QUEUE.pop(0)
        return None

async def _queue_position(task_id: str) -> Optional[int]:
    async with QUEUE_LOCK:
        try:
            idx = QUEUE.index(task_id)
            return idx + 1
        except ValueError:
            return None

def _make_task(task_id: str, file_path: str):
    TASKS[task_id] = {
        "task_id": task_id,
        "file_path": file_path,
        "status": "PENDING",
        "results": None,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }

def _validate_file_exists(file_path: str):
    if not os.path.isabs(file_path):
        raise HTTPException(status_code=400, detail="Ожидается абсолютный путь к файлу.")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден.")
    if not _ext_ok(file_path):
        raise HTTPException(status_code=415, detail="Неподдерживаемый формат аудио.")



def _transcribe_blocking(file_path: str) -> List[Dict[str, Any]]:

    if MODEL_IMPL == "faster_whisper":
        segments, _info = WHISPER.transcribe(
            file_path,
            vad_filter=True,
            beam_size=1,
        )
        phrases = []
        for seg in segments:
            phrases.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })
        return phrases

    if MODEL_IMPL == "openai_whisper":
        result = WHISPER.transcribe(file_path, task="transcribe", verbose=False)
        phrases = []
        for seg in result.get("segments", []):
            phrases.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            })
        return phrases

    basename = os.path.basename(file_path)
    return [
        {"start": 0.0, "end": 2.5, "text": f"Демонстрационный результат для файла {basename}."},
        {"start": 2.7, "end": 5.2, "text": "Установите ffmpeg и библиотеку Whisper для реальной транскрибации."},
    ]


async def _transcribe_async(file_path: str) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_blocking, file_path)



async def _worker_loop():
    while True:
        task_id = await _queue_pop_left()
        if not task_id:
            await asyncio.sleep(0.2)
            continue

        task = TASKS.get(task_id)
        if not task:
            continue

        task["status"] = "IN_PROGRESS"
        task["started_at"] = _now_iso()

        try:
            phrases = await _transcribe_async(task["file_path"])
            task["results"] = {"phrases": phrases}
            task["status"] = "COMPLETED"
        except Exception as e:
            task["error"] = str(e)
            task["results"] = None
            task["status"] = "COMPLETED"
        finally:
            task["finished_at"] = _now_iso()




@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in SUPPORTED_EXT:
        raise HTTPException(status_code=415, detail="Неподдерживаемый формат аудио.")

    uploads_dir = os.path.abspath("./uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    dst_path = os.path.join(uploads_dir, f"{uuid.uuid4().hex}{ext}")

    with open(dst_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"file_path": dst_path}


@app.post("/api/v1/create_task")
async def create_task(payload: CreateTaskIn) -> dict:


    _validate_file_exists(payload.file_path)
    task_id = uuid.uuid4().hex
    _make_task(task_id, payload.file_path)
    await _queue_push(task_id)
    return {"task_id": task_id}


@app.get("/api/v1/status/{task_id}")
async def get_status(
    task_id: str,
    details: bool = Query(False, description="Если true — добавим позицию в очереди и временные метки"),
) -> dict:

    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    status: Status = task["status"]

    base = {"task_id": task_id, "status": status}

    if status == "COMPLETED":
        if task["results"] is None and task.get("error"):

            pass
        else:
            base["results"] = task["results"]


    if details:
        base["_details"] = {
            "queue_position": await _queue_position(task_id),
            "created_at": task["created_at"],
            "started_at": task["started_at"],
            "finished_at": task["finished_at"],
            "error": task.get("error"),
            "model_impl": MODEL_IMPL,
            "model_name": WHISPER_MODEL_NAME,
        }

    return base
