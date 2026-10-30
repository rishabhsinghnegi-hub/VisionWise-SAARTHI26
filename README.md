# Vision Wise

Vision Wise is a visual assistance prototype using an ESP32-CAM as its only camera. Its Wi-Fi MJPEG stream is displayed in the Next.js app while FastAPI reads frames from the stream and runs YOLO11n. The API returns detections and normalized boxes. The browser draws those boxes over the matching camera view, sends only current detection metadata to the local Ollama assistant, and reads responses through system speech synthesis.

## Current model and limits

The repository contains `yolo11n.pt` at the project root. There is no `best.pt` or project-specific fine-tune in this checkout. This is the pretrained YOLO11n COCO model, which recognizes its 80 trained COCO classes; it is not a custom-trained Vision Wise model. Vision Wise uses the existing weights without retraining or silently downloading a replacement. Set `VISIONWISE_MODEL_PATH` to a compatible trained weights file if one is supplied later.

All boxes come from YOLO's individual normalized bounding boxes and are sorted left to right. Left and right refer to the image. The ESP32-CAM does not measure physical distance or determine which person is nearest, so the assistant explicitly says when that cannot be known. Detection is fallible and is not an obstacle-avoidance or emergency-safety system.

## Architecture

- `apps/web`: Next.js client with a shared MJPEG preview, YOLO box overlays, voice controls, settings, and local detection history.
- `apps/api`: FastAPI reads the ESP32-CAM stream once and shares frames between YOLO inference and the live preview, alongside deterministic visual-intent routing, settings, and SQLite summaries.
- `apps/api/inference.py`: loads the configured YOLO weights once, reuses the model, and uses CUDA when supported or CPU otherwise.
- Ollama: optional local language model for general conversational questions. Deterministic visual answers continue to work without Ollama.
- Browser audio: explicit wake-phrase activation and push-to-talk use the browser Web Speech recognition API; spoken replies use browser speech synthesis.

No raw camera frames or audio are saved. Only selected detection summaries and metadata are stored when history is enabled. Ollama receives text and current structured detections, never images. **Browser speech recognition may send audio to the browser vendor's speech service.** I checked [openWakeWord's documented model support](https://github.com/dscripka/openWakeWord): it supplies phrase-specific models and a custom-training path, but this repository has no verified “Vision Wise” wake model or local audio service. This version therefore matches the configurable phrase in browser recognition instead of claiming an offline wake detector; recognition can be unreliable across browsers. Push-to-talk is available as a fallback.

## Install and run

Requirements: Node.js 20.9+, Python 3.10+, the repository's `yolo11n.pt`, and optionally [Ollama](https://ollama.com).

Pull the lightweight local chat model once, in a terminal with Ollama installed:

```powershell
ollama pull qwen2.5:3b
```

Keep the Ollama app/service running. On systems where it is not already running, use `ollama serve` in its own terminal.

In a second terminal, install and start the API:

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set ESP32_STREAM_URL to the camera's LAN URL.
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the website:

```powershell
cd apps/web
npm install
npm.cmd run dev
```

Open <http://localhost:3000>. Select **Start ESP32 Camera** to verify the camera stream and begin automatic detection. Set `ESP32_STREAM_URL` in `apps/api/.env` to `http://<esp32-ip>:81/stream` for your network. Select **Activate “Vision Wise” wake phrase** or **Push to talk** to explicitly activate browser microphone recognition. The first activation prompts for microphone permission. Stop camera and voice separately with their controls. Browser speech recognition support varies; Chrome and Edge usually expose the API. Use the text chat if voice recognition is unavailable.

## Local configuration

- `VISIONWISE_MODEL_PATH`: compatible YOLO weights path. Default: repository-root `yolo11n.pt`.
- `VISIONWISE_DB_PATH`: SQLite database path. Default: `apps/api/visionwise.db`.
- `OLLAMA_BASE_URL`: local Ollama API URL. Default: `http://127.0.0.1:11434`.
- `OLLAMA_MODEL`: installed local assistant model. Default: `qwen2.5:3b`.
- `ESP32_STREAM_URL`: ESP32-CAM MJPEG stream URL. Set this in `apps/api/.env`; there is no universal default because the camera IP depends on the local network.

No paid API key is required or used by the assistant. The API's `/health` endpoint reports YOLO status and whether Ollama is running and the configured model is installed. Supported visual questions such as scene descriptions, object presence, side-of-image queries, and nearest-person questions are answered deterministically without Ollama.

## Routes

- `GET /health`: API, YOLO, and Ollama model status.
- `GET /api/esp32/detect?confidence=0.25&mode=everyday&save_history=false`: read an ESP32-CAM frame and run detection.
- `GET /api/esp32/stream`: browser preview of the shared ESP32-CAM MJPEG reader.
- `POST /api/esp32/stop`: stop the shared reader when the user stops the camera.
- `POST /api/detect`: JPEG, PNG, or WebP frame inference.
- `POST /api/assistant`: current detection-grounded question routing and optional local Ollama chat.
- `GET /api/history`: stored detection summaries.
- `DELETE /api/history`: clear saved detection history.
- `GET /api/stats`: event counts and mean inference latency.
- `GET /settings`, `PUT /settings`: confidence, mode, privacy, narration, wake phrase, and speech settings.

ESP32-CAM frames are processed in memory. FastAPI keeps one MJPEG connection and shares its latest frames between YOLO inference and the frontend preview. The browser does not open a camera stream directly or capture frames.

## Verification and manual checks

Run the frontend type, lint, and production build checks, plus the API syntax check:

```powershell
cd apps/web
.\node_modules\.bin\tsc.cmd --noEmit
npm.cmd run lint
npm.cmd run build
cd ..\api
python -m py_compile main.py inference.py rules.py esp32_camera.py
```

Manual end-to-end checks:

1. Start Ollama, API, and web app; verify the UI reports YOLO and Ollama model status.
2. Confirm the ESP32-CAM stream is reachable, start the camera, and place multiple COCO objects in view. Confirm each label and box tracks its object, confidence is shown, and labels are ordered left to right.
3. Ask “describe my surroundings,” “is there a person on my left,” and “who is the nearest person?” Verify image-relative wording and that the assistant never claims physical distance.
4. Activate the wake phrase and say “Vision Wise” followed by a question. Also try push-to-talk, typed chat, repeat, stop-speaking, and camera/voice deactivation.
5. Disable Ollama and repeat the supported visual questions; deterministic answers should still work. General chat should show a clear local-service error.
6. Save a detection, refresh history, then use **Clear all** and confirm it is empty.
7. Test keyboard focus, screen-reader state announcements, reduced-motion settings, and mobile layout.

Camera, microphone, speech recognition, speaker output, Ollama response quality/performance, and screen-reader behavior depend on local hardware/browser and were not verified by automated checks in this workspace. Browser wake-phrase recognition is not a robust offline wake-word engine. The only available weight file is COCO-pretrained YOLO11n, not project-specific trained weights.
