"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

const API = "/vision-api";

type Detection = { label: string; confidence: number; box: number[]; position: string };
type Result = {
  detections: Detection[];
  guidance: { summary: string; relevant_count: number; relevant_detections: Detection[]; rule: string; notice: string };
  inference_ms: number;
  model: { name: string; device: string };
  ai_message?: string;
};
type EventItem = { id: number; created_at: string; mode: string; summary: string; detections: Detection[]; inference_ms: number };
type Health = { ok: boolean; model: { available: boolean; name: string; device: string; error: string | null; class_count?: number | null; weights?: string }; assistant?: { provider: string; model: string; local: boolean; available: boolean; installed: boolean }; hardware: { distance_sensor: string } };
type Settings = { confidence: number; mode: "everyday" | "navigation" | "indoor"; save_history: boolean; wake_phrase: string; continuous_narration: boolean; speech_rate: number; speech_volume: number; speech_voice: string };
type ChatMessage = { role: "user" | "assistant"; content: string };
type TranscriptEntry = { id: number; kind: "you" | "assistant" | "vision"; text: string; time: string };
type CameraSource = "esp32" | null;
type SpeechRecognizer = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: Event) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};
type SpeechRecognitionResultEvent = Event & { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string; isFinal: boolean }>> };
type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognizer;
  webkitSpeechRecognition?: new () => SpeechRecognizer;
};

const defaultSettings: Settings = { confidence: 0.2, mode: "everyday", save_history: true, wake_phrase: "Vision Wise", continuous_narration: false, speech_rate: 1, speech_volume: 0.85, speech_voice: "" };
const nav = [
  { id: "live", label: "Live Assistance", icon: "◉" },
  { id: "history", label: "Detection History", icon: "◷" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

function speak(text: string, rate: number, volume: number, voiceName = "", onEnd?: () => void) {
  if (!("speechSynthesis" in window)) return false;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = rate;
  utterance.volume = volume;
  utterance.voice = window.speechSynthesis.getVoices().find((voice) => voice.name === voiceName) ?? null;
  utterance.onend = onEnd ?? null;
  utterance.onerror = onEnd ? () => onEnd() : null;
  window.speechSynthesis.speak(utterance);
  return true;
}

export default function Home() {
  const busyRef = useRef(false);
  const assistantBusyRef = useRef(false);
  const speechEnabledRef = useRef(true);
  const voiceModeRef = useRef<"off" | "wake" | "push">("off");
  const pauseWakeRef = useRef(false);
  const lastNarratedRef = useRef("");
  const lastLoggedDetectionRef = useRef("");
  const lastSpokenSceneAtRef = useRef(0);
  const lastSafetyAlertAtRef = useRef(0);
  const lastSceneLoggedAtRef = useRef(0);
  const pendingSceneAnnouncementRef = useRef("");
  const wakePhraseRef = useRef(defaultSettings.wake_phrase);
  const awaitingWakeQuestionRef = useRef(false);
  const cameraSourceRef = useRef<CameraSource>(null);
  const lastResponseRef = useRef("");
  const speechSequenceRef = useRef(0);
  const detectionsRef = useRef<Detection[]>([]);
  const cameraActiveRef = useRef(false); 
  const hasSnapshotRef = useRef(false);
  const snapshotAtRef = useRef(0);
  const chatMessagesRef = useRef<ChatMessage[]>([]);
  const speechVoiceRef = useRef("");
  const recognitionRef = useRef<SpeechRecognizer | null>(null);
  const startWakeListeningRef = useRef<() => void>(() => undefined);
  const sayRef = useRef<(text: string, resumeWake?: boolean) => void>(() => undefined);
  const [active, setActive] = useState("live");
  const [cameraOn, setCameraOn] = useState(false);
  const [hasSnapshot, setHasSnapshot] = useState(false);
  const [cameraSource, setCameraSource] = useState<CameraSource>(null);
  const [cameraError, setCameraError] = useState("");
  const [apiError, setApiError] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [settings, setSettings] = useState(defaultSettings);
  const [result, setResult] = useState<Result | null>(null);
  const [history, setHistory] = useState<EventItem[]>([]);
  const [query, setQuery] = useState("");
  const [eventsToday, setEventsToday] = useState(0);
  const [speechOn, setSpeechOn] = useState(true);
  const [speechRate, setSpeechRate] = useState(defaultSettings.speech_rate);
  const [speechVolume, setSpeechVolume] = useState(defaultSettings.speech_volume);
  const [busy, setBusy] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [sessionTranscript, setSessionTranscript] = useState<TranscriptEntry[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatError, setChatError] = useState("");
  const [voiceListening, setVoiceListening] = useState(false);
  const [voiceMode, setVoiceMode] = useState<"off" | "wake" | "push">("off");
  const [voiceStatus, setVoiceStatus] = useState("Voice assistant off");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastResponse, setLastResponse] = useState("");
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [speechVoice, setSpeechVoice] = useState("");

  const loadHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API}/health`, { cache: "no-store" });
      if (!response.ok) throw new Error("API not ready");
      setHealth(await response.json());
      setApiError("");
    } catch {
      setHealth(null);
      setApiError("The local VisionWise API is not reachable. Start the API to enable camera analysis.");
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/history?limit=50&q=${encodeURIComponent(query)}`, { cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      setHistory(data.items);
    } catch { /* the system status communicates API availability */ }
  }, [query]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void loadHealth();
      void loadHistory();
      fetch(`${API}/api/stats`).then((r) => r.ok ? r.json() : null).then((s) => { if (s) setEventsToday(s.events_today); }).catch(() => undefined);
      fetch(`${API}/settings`).then((r) => r.ok ? r.json() : defaultSettings).then((s: Settings) => { setSettings(s); setSpeechRate(s.speech_rate); setSpeechVolume(s.speech_volume); setSpeechVoice(s.speech_voice); }).catch(() => undefined);
    }, 0);
    const timer = window.setInterval(() => { void loadHealth(); }, 10000);
    return () => { window.clearTimeout(initialLoad); window.clearInterval(timer); };
  }, [loadHealth, loadHistory]);

  const appendTranscript = useCallback((kind: TranscriptEntry["kind"], text: string) => {
    setSessionTranscript((entries) => [...entries, {
      id: Date.now() + Math.random(), kind, text,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    }]);
  }, []);

  const recordScene = useCallback((detections: Detection[]) => {
    const signature = detections.map((item) => {
      const area = (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]);
      const sizeBand = area >= 0.35 ? "large" : area >= 0.2 ? "medium" : area >= 0.08 ? "small" : "tiny";
      return `${item.label}:${sizeBand}`;
    }).sort().join("|") || "empty";
    if (signature === lastLoggedDetectionRef.current) return null;
    if (Date.now() - lastSceneLoggedAtRef.current < 2500) return null;
    lastLoggedDetectionRef.current = signature;
    lastSceneLoggedAtRef.current = Date.now();
    const message = detections.length
      ? `Detected ${detections.length} ${detections.length === 1 ? "object" : "objects"}: ${detections.map((item) => `${item.label} (${item.position})`).join(", ")}.`
      : "No recognizable objects in the latest camera frame.";
    appendTranscript("vision", message);
    return message;
  }, [appendTranscript]);

  const say = useCallback((text: string, resumeWake = false) => {
    const sequence = ++speechSequenceRef.current;
    if (voiceModeRef.current === "wake") {
      pauseWakeRef.current = true;
      recognitionRef.current?.stop();
    }
    setIsSpeaking(true);
    setVoiceStatus("Speaking");
    const finished = () => {
      if (sequence !== speechSequenceRef.current) return;
      setIsSpeaking(false);
      if (resumeWake && voiceModeRef.current === "wake" && pendingSceneAnnouncementRef.current) {
        const pending = pendingSceneAnnouncementRef.current;
        pendingSceneAnnouncementRef.current = "";
        lastSpokenSceneAtRef.current = Date.now();
        window.setTimeout(() => sayRef.current(pending, true), 120);
        return;
      }
      if (resumeWake && voiceModeRef.current === "wake") {
        pauseWakeRef.current = false;
        setVoiceStatus(awaitingWakeQuestionRef.current ? "Listening for your question" : `Listening for wake phrase: ${wakePhraseRef.current}`);
        startWakeListeningRef.current();
      } else if (voiceModeRef.current === "wake") {
        pauseWakeRef.current = false;
      } else {
        setVoiceStatus("Response spoken");
      }
    };
    if (!speak(text, speechRate, speechVolume, speechVoiceRef.current, finished)) finished();
  }, [speechRate, speechVolume]);

  useEffect(() => { sayRef.current = say; }, [say]);

  const announceScene = useCallback((detections: Detection[], message: string | null) => {
    if (!message || !detections.length || voiceModeRef.current !== "wake" || !speechEnabledRef.current) return;
    const now = Date.now();
    const largeLowerObject = detections.find((item) => {
      const [x1, y1, x2, y2] = item.box;
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      return (x2 - x1) * (y2 - y1) >= 0.35 && centerX >= 0.15 && centerX <= 0.85 && centerY >= 0.55;
    });
    if (largeLowerObject) {
      if (now - lastSafetyAlertAtRef.current < 12000) return;
      lastSafetyAlertAtRef.current = now;
    } else return;
    const spoken = largeLowerObject
      ? `Caution. A ${largeLowerObject.label} fills much of the lower camera view. This is not a collision prediction. I cannot measure distance; please pause and check your surroundings.`
      : message;
    if (assistantBusyRef.current || window.speechSynthesis?.speaking) {
      pendingSceneAnnouncementRef.current = spoken;
      return;
    }
    lastSpokenSceneAtRef.current = now;
    appendTranscript("assistant", spoken);
    say(spoken, true);
  }, [appendTranscript, say]);

  useEffect(() => {
    const updateVoices = () => setAvailableVoices(window.speechSynthesis?.getVoices() ?? []);
    updateVoices();
    window.speechSynthesis?.addEventListener("voiceschanged", updateVoices);
    return () => window.speechSynthesis?.removeEventListener("voiceschanged", updateVoices);
  }, []);

  useEffect(() => { speechVoiceRef.current = speechVoice; }, [speechVoice]);
  useEffect(() => { cameraSourceRef.current = cameraSource; }, [cameraSource]);
  useEffect(() => { wakePhraseRef.current = settings.wake_phrase.trim() || defaultSettings.wake_phrase; }, [settings.wake_phrase]);

  const saveSettings = useCallback(async (next: Settings) => {
    setSettings(next);
    try {
      await fetch(`${API}/settings`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next) });
    } catch { /* settings remain available for this session */ }
  }, []);

  const startCamera = async () => {
    setCameraError("");
    setApiError("");
    try {
      const healthResponse = await fetch(`${API}/health`, { cache: "no-store" });
      if (!healthResponse.ok) throw new Error("The VisionWise backend is not ready. Start the API at http://127.0.0.1:8000 and try again.");
      const params = new URLSearchParams({ confidence: String(settings.confidence), mode: settings.mode, save_history: String(settings.save_history) });
      const response = await fetch(`${API}/api/esp32/detect?${params}`, { cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `ESP32-CAM capture failed (HTTP ${response.status}). Check ESP32_STREAM_URL in apps/api/.env.`);
      }
      await response.json();
      detectionsRef.current = [];
      setResult(null);
      setHasSnapshot(false);
      hasSnapshotRef.current = false;
      snapshotAtRef.current = 0;
      setLastUpdated("");
      setCameraOn(true);
      setCameraSource("esp32");
      cameraActiveRef.current = true;
      lastLoggedDetectionRef.current = "";
      lastNarratedRef.current = "";
      setActive("live");
      setApiError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not connect to the configured ESP32-CAM stream.";
      setCameraError(message);
      cameraActiveRef.current = false;
      setCameraOn(false);
      setCameraSource(null);
    }
  };

  const stopCamera = () => {
    cameraActiveRef.current = false;
    setCameraOn(false);
    setCameraSource(null);
    window.speechSynthesis?.cancel();
    speechSequenceRef.current += 1;
    setIsSpeaking(false);
    void fetch(`${API}/api/esp32/stop`, { method: "POST", cache: "no-store" }).catch(() => undefined);
  };

  const analyze = useCallback(async () => {
    if (!cameraActiveRef.current || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      const params = new URLSearchParams({ confidence: String(settings.confidence), mode: settings.mode, save_history: String(settings.save_history) });
      const response = await fetch(`${API}/api/esp32/detect?${params}`, { cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `ESP32-CAM detection failed (HTTP ${response.status}). Check the camera stream and backend.`);
      }
      const data: Result = await response.json();
      data.detections.sort((a, b) => (a.box[0] + a.box[2]) - (b.box[0] + b.box[2]));
      detectionsRef.current = data.detections;
      const sceneMessage = recordScene(data.detections);
      announceScene(data.detections, sceneMessage);
      hasSnapshotRef.current = true;
      setHasSnapshot(true);
      snapshotAtRef.current = Date.now();
      setResult(data);
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      setApiError("");
      if (settings.continuous_narration && voiceModeRef.current === "off" && data.detections.length && data.guidance.summary !== lastNarratedRef.current) {
        lastNarratedRef.current = data.guidance.summary;
        if (speechEnabledRef.current) say(data.guidance.summary);
      }
      if (settings.save_history) {
        void loadHistory();
        fetch(`${API}/api/stats`).then((r) => r.ok ? r.json() : null).then((s) => { if (s) setEventsToday(s.events_today); }).catch(() => undefined);
      }
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "ESP32-CAM analysis failed.");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [settings, loadHistory, recordScene, say, announceScene]);

  const startWakeListening = () => {
    startWakeListeningRef.current = startWakeListening;
    if (voiceModeRef.current !== "wake" || pauseWakeRef.current) return;
    const speechWindow = window as SpeechRecognitionWindow;
    const Recognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const resultEvent = event as SpeechRecognitionResultEvent;
      const finalText = Array.from({ length: resultEvent.results.length - resultEvent.resultIndex }, (_, index) => resultEvent.results[resultEvent.resultIndex + index])
        .filter((item) => item[0]?.isFinal)
        .map((item) => item[0]?.transcript ?? "")
        .join(" ")
        .trim();
      if (!finalText) return;
      const command = finalText.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
      const startCameraCommand = /^(please )?(can you )?(?:(start|open|activate|turn on)( the)?( esp32( cam)?| camera)?|turn (the )?camera on)( please)?$/.test(command);
      const stopCameraCommand = /^(please )?(can you )?(?:(stop|close|deactivate|turn off)( the)?( esp32( cam)?| camera)?|turn (the )?camera off)( please)?$/.test(command);
      if (startCameraCommand) {
        appendTranscript("you", finalText);
        if (cameraActiveRef.current) {
          say("The camera is already running.", true);
        } else {
          document.querySelector<HTMLButtonElement>("[data-camera-start]")?.click();
          const message = "Starting the ESP32-CAM.";
          appendTranscript("assistant", message);
          say(message, true);
        }
        return;
      }
      if (stopCameraCommand) {
        appendTranscript("you", finalText);
        if (cameraSourceRef.current === null) {
          say("The camera is already off. The voice assistant is still on.", true);
        } else {
          document.querySelector<HTMLButtonElement>("[data-camera-stop]")?.click();
          const message = "Camera stopped. The voice assistant is still on. Ask me a question or say start camera.";
          appendTranscript("assistant", message);
          say(message, true);
        }
        return;
      }
      const wakePhrase = wakePhraseRef.current.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
      const normalizedText = finalText.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
      const wakeIndex = normalizedText.indexOf(wakePhrase);
      if (awaitingWakeQuestionRef.current) {
        awaitingWakeQuestionRef.current = false;
        setChatInput(finalText);
        void askAssistantQuestion(finalText);
      } else if (wakePhrase && wakeIndex >= 0) {
        const question = `${normalizedText.slice(0, wakeIndex)} ${normalizedText.slice(wakeIndex + wakePhrase.length)}`.trim();
        if (question) {
          setChatInput(question);
          void askAssistantQuestion(question);
        } else {
          awaitingWakeQuestionRef.current = true;
          appendTranscript("you", finalText);
          say("Yes, I'm listening. What would you like to know?", true);
        }
      } else {
        setVoiceStatus(`Listening for the wake phrase: ${wakePhraseRef.current}`);
      }
    };
    recognition.onerror = (event) => {
      const code = (event as Event & { error?: string }).error;
      if (["not-allowed", "service-not-allowed", "audio-capture", "network"].includes(code ?? "")) {
        voiceModeRef.current = "off";
        setVoiceMode("off");
        const message = code === "not-allowed" || code === "service-not-allowed"
          ? "Microphone permission denied. Allow microphone access in browser settings."
          : "Browser speech service unavailable. Use push-to-talk or text chat.";
        setVoiceStatus(message);
        if (speechEnabledRef.current) say(message);
      } else if (code !== "no-speech") {
        setVoiceStatus("Microphone recognition interrupted; reconnecting");
      }
    };
    recognition.onend = () => {
      setVoiceListening(false);
      if (voiceModeRef.current === "wake" && !pauseWakeRef.current) {
        window.setTimeout(() => startWakeListening(), 450);
      }
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setVoiceListening(true);
      setVoiceStatus(awaitingWakeQuestionRef.current ? "Listening for your question" : `Listening for wake phrase: ${wakePhraseRef.current}`);
    } catch {
      setVoiceStatus("Could not start microphone recognition");
    }
  };

  const askAssistantQuestion = async (rawMessage: string) => {
    const message = rawMessage.trim();
    if (!message || assistantBusyRef.current) return;
    assistantBusyRef.current = true;
    const nextMessages = [...chatMessagesRef.current, { role: "user" as const, content: message }];
    chatMessagesRef.current = nextMessages;
    setChatMessages(nextMessages);
    appendTranscript("you", message);
    setChatInput("");
    setChatBusy(true);
    setChatError("");
    setVoiceStatus("Processing your question");
    if (voiceModeRef.current === "wake") {
      pauseWakeRef.current = true;
      recognitionRef.current?.stop();
    }
    try {
      const response = await fetch(`${API}/api/assistant`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          detections: detectionsRef.current,
          vision_available: cameraActiveRef.current && hasSnapshotRef.current,
          history: chatMessagesRef.current.slice(-12, -1),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "The local assistant could not reply.");
      const reply = String(data.reply);
      const next = [...nextMessages, { role: "assistant" as const, content: reply }];
      chatMessagesRef.current = next;
      setChatMessages(next);
      appendTranscript("assistant", reply);
      lastResponseRef.current = reply;
      setLastResponse(reply);
      if (speechOn) say(reply, voiceModeRef.current === "wake");
      else if (voiceModeRef.current === "wake") {
        pauseWakeRef.current = false;
        startWakeListening();
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "The local assistant could not reply.";
      setChatError(message);
      appendTranscript("assistant", `I couldn't answer: ${message}`);
      if (voiceModeRef.current === "wake") {
        say("Sorry, I couldn't process that question. Please try again.", true);
      }
    } finally {
      assistantBusyRef.current = false;
      setChatBusy(false);
    }
  };

  const sendAssistantMessage = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void askAssistantQuestion(chatInput);
  };

  const stopSpeaking = () => {
    speechSequenceRef.current += 1;
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
    if (voiceModeRef.current === "wake") {
      pauseWakeRef.current = false;
      startWakeListening();
    } else setVoiceStatus("Speech stopped");
  };

  const repeatLastResponse = () => {
    if (lastResponseRef.current) say(lastResponseRef.current, voiceModeRef.current === "wake");
  };

  const toggleWakeAssistant = () => {
    if (voiceModeRef.current === "wake") {
      voiceModeRef.current = "off";
      pauseWakeRef.current = false;
      awaitingWakeQuestionRef.current = false;
      recognitionRef.current?.stop();
      setVoiceMode("off");
      setVoiceListening(false);
      setVoiceStatus("Voice assistant off");
      pendingSceneAnnouncementRef.current = "";
      speechSequenceRef.current += 1;
      window.speechSynthesis?.cancel();
      setIsSpeaking(false);
      if (speechOn) say("Voice assistant off.");
      return;
    }
    const speechWindow = window as SpeechRecognitionWindow;
    if (!(speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition)) {
      const message = "Continuous voice input is not supported in this browser. Use push-to-talk or type your question.";
      setChatError(message);
      if (speechEnabledRef.current) say(message);
      return;
    }
    recognitionRef.current?.stop();
    pauseWakeRef.current = false;
    window.speechSynthesis?.cancel();
    voiceModeRef.current = "wake";
    pauseWakeRef.current = false;
    setVoiceMode("wake");
    startWakeListening();
    awaitingWakeQuestionRef.current = false;
    if (speechOn) say(`Voice assistant is on. Say ${wakePhraseRef.current} before your question. You can say start camera or stop camera at any time. Press Alt Shift C to toggle the camera, or Alt Shift V to turn the assistant off. I can describe objects, but I cannot measure distance or predict a collision.`, true);
  };

  const togglePushToTalk = () => {
    if (voiceModeRef.current === "push") {
      recognitionRef.current?.stop();
      voiceModeRef.current = "off";
      setVoiceMode("off");
      setVoiceStatus("");
      return;
    }
    const speechWindow = window as SpeechRecognitionWindow;
    const Recognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Recognition) {
      setChatError("Voice input is unavailable in this browser. Try Chrome or Edge, or type your question.");
      return;
    }
    recognitionRef.current?.stop();
    pauseWakeRef.current = false;
    const recognition = new Recognition();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const resultEvent = event as SpeechRecognitionResultEvent;
      setChatInput(Array.from(resultEvent.results, (item) => item[0]?.transcript ?? "").join(" ").trim());
    };
    recognition.onerror = (event) => {
      const code = (event as Event & { error?: string }).error;
      setChatError(code === "not-allowed" || code === "service-not-allowed" ? "Microphone permission was denied. Allow it in browser settings and try again." : "Speech recognition stopped. Check microphone permission and browser speech service, then try again.");
    };
    recognition.onend = () => {
      setVoiceListening(false);
      if (voiceModeRef.current === "push") {
        voiceModeRef.current = "off";
        setVoiceMode("off");
      }
      setVoiceStatus("Question ready to send");
    };
    recognitionRef.current = recognition;
    voiceModeRef.current = "push";
    setVoiceMode("push");
    setVoiceStatus("Listening for your question");
    try {
      recognition.start();
      setVoiceListening(true);
    } catch {
      setChatError("Could not start microphone recognition. Check permission and try again.");
    }
  };

  const clearHistory = async () => {
    try {
      const response = await fetch(`${API}/api/history`, { method: "DELETE" });
      if (!response.ok) throw new Error("Could not clear detection history.");
      setHistory([]);
      setEventsToday(0);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Could not clear detection history.");
    }
  };

  useEffect(() => {
    const onShortcut = (event: KeyboardEvent) => {
      if (!event.altKey || !event.shiftKey || event.ctrlKey || event.metaKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || target.matches("input, textarea, select"))) return;
      const key = event.key.toLowerCase();
      if (key === "v") {
        event.preventDefault();
        document.querySelector<HTMLButtonElement>("[data-voice-toggle]")?.click();
      } else if (key === "c") {
        event.preventDefault();
        const stopButton = document.querySelector<HTMLButtonElement>("[data-camera-stop]");
        const startButton = document.querySelector<HTMLButtonElement>("[data-camera-start]");
        (stopButton ?? startButton)?.click();
      }
    };
    window.addEventListener("keydown", onShortcut);
    return () => window.removeEventListener("keydown", onShortcut);
  }, []);

  useEffect(() => {
  if (!cameraOn || cameraSource !== "esp32") return;

    const timer = window.setInterval(() => {
      void analyze();
  }, 2200);

  return () => window.clearInterval(timer);
}, [cameraOn, cameraSource, analyze]);

  useEffect(() => () => {
    recognitionRef.current?.stop();
    window.speechSynthesis?.cancel();
    void fetch(`${API}/api/esp32/stop`, { method: "POST", keepalive: true }).catch(() => undefined);
  }, []);

  const navigate = (id: string) => {
    setActive(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (id === "history") void loadHistory();
  };

  const totalDetections = result?.detections.length ?? 0;
  const activePage = nav.find((item) => item.id === active)?.label ?? "Live Assistance";

  return (
    <main className="shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className="sidebar" aria-label="Main navigation">
        <a href="#live" className="brand" onClick={() => setActive("live")}>
          <span className="brand-mark" aria-hidden="true">V</span>
          <span className="brand-text"><strong>Vision Wise</strong><small>VISUAL ASSISTANCE</small></span>
        </a>
        <p className="nav-caption">WORKSPACE</p>
        <nav className="nav-list" aria-label="Workspace">
          {nav.map((item) => <button key={item.id} className={`nav-item ${active === item.id ? "selected" : ""}`} onClick={() => navigate(item.id)} aria-current={active === item.id ? "page" : undefined}>
            <span className="nav-icon" aria-hidden="true">{item.icon}</span>{item.label}
            {item.id === "live" && cameraOn && <span className="nav-live-dot" aria-label="Camera active" />}
          </button>)}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-card"><span className="privacy-icon" aria-hidden="true">◈</span><strong>Private by design</strong><p>Frames are analyzed locally and never saved. Only selected detection summaries enter your history.</p></div>
          <span className="version-label">LOCAL AI · EARLY ACCESS</span>
        </div>
      </aside>

      <section className="main-column" id="main-content">
        <header className="topbar">
          <div><p className="breadcrumb">Workspace <span>/</span> {activePage}</p><h1>{activePage}</h1></div>
          <div className="topbar-right"><span className={`connection-pill ${health?.ok ? "connected" : ""}`}><span className="connection-dot" />{health?.ok ? "API connected" : "API offline"}</span><span className="avatar" aria-label="Vision Wise workspace">VW</span></div>
        </header>

        <div className="content" id="overview">
          <section className="welcome-card">
            <div className="welcome-copy"><p className="kicker">LOCAL AI · ESP32-CAM · YOUR CHOICE</p><h2>Understand a little more<br className="desktop-break" /> of what’s around you.</h2><p>Use the ESP32-CAM stream to hear and review concise object descriptions, processed locally by VisionWise.</p><button className="primary-button" onClick={() => cameraOn ? navigate("live") : void startCamera()}>{cameraOn ? "View live camera" : "Start ESP32 Camera"}<span aria-hidden="true">→</span></button><span className="hero-footnote"><span className="tiny-lock" aria-hidden="true">◈</span> Camera starts only when you choose</span></div>
            <div className="hero-visual" aria-hidden="true"><div className="hero-ring ring-one"/><div className="hero-ring ring-two"/><div className="hero-ring ring-three"/><div className="hero-center"><span>V</span><i/></div><span className="orbit-point point-one"/><span className="orbit-point point-two"/></div>
            <div className="hero-index" aria-hidden="true">01 <span>/</span> LOCAL VISION</div>
          </section>

          <section className="metrics-grid" aria-label="System summary">
            <article className="metric-card"><div className="metric-heading"><span>Camera</span><span className="metric-icon">◎</span></div><strong>{cameraOn ? "Streaming" : "Off"}</strong><small>{cameraOn ? "ESP32-CAM · MJPEG stream" : "ESP32-CAM ready to connect"}</small></article>
            <article className="metric-card"><div className="metric-heading"><span>Detection model</span><span className="metric-icon">✳</span></div><strong className="metric-model">{health?.model.name ?? "YOLO11n"}</strong><small>{health?.model.available ? `Running on ${health.model.device}` : "Waiting for local model"}</small></article>
            <article className="metric-card"><div className="metric-heading"><span>Objects in this frame</span><span className="metric-icon">◉</span></div><strong>{totalDetections}</strong><small>{lastUpdated ? `Updated at ${lastUpdated}` : "No frame analyzed yet"}</small></article>
            <article className="metric-card"><div className="metric-heading"><span>Events today</span><span className="metric-icon">◷</span></div><strong>{eventsToday}</strong><small>Selected summaries saved locally</small></article>
          </section>

          <section className="workspace-grid" id="live">
            <article className="panel camera-panel">
              <div className="panel-heading"><div><p className="eyebrow">ESP32-CAM WORKSPACE</p><h2>Live camera</h2></div><span className={`live-status ${cameraOn && cameraSource === "esp32" ? "is-live" : ""}`}><i />{cameraOn ? "ESP32-CAM live" : "ESP32-CAM standby"}</span></div>
              <div className="camera-source-controls" aria-label="ESP32-CAM controls">
                {!cameraOn && <button type="button" data-camera-start aria-keyshortcuts="Alt+Shift+C" className="secondary-button" onClick={() => void startCamera()} disabled={busy}>{busy ? "Connecting…" : "Start ESP32 Camera"}</button>}
                {cameraOn && <button type="button" data-camera-stop aria-keyshortcuts="Alt+Shift+C" className="text-button" onClick={stopCamera}>Stop Camera</button>}
              </div>
              <div className={`camera-stage ${cameraOn && cameraSource === "esp32" ? "camera-running" : ""}`}>
                {/* eslint-disable-next-line @next/next/no-img-element -- The shared FastAPI MJPEG route must be rendered as an img stream. */}
                {cameraOn && cameraSource === "esp32" && <img src={`${API}/api/esp32/stream`} alt="ESP32-CAM live stream" className="esp32-camera-feed" />}
                {!cameraOn && <div className="camera-empty"><div className="camera-glyph">◎</div><strong>ESP32-CAM is ready</strong><p>Start the camera to connect to the live stream and run YOLO detection.</p></div>}
                {cameraOn && result && hasSnapshot && <div className="detection-overlay" aria-label="Detected objects">{result.detections.map((d, index) => <div key={`${d.label}-${index}`} className="detection-box" style={{ left: `${d.box[0] * 100}%`, top: `${d.box[1] * 100}%`, width: `${(d.box[2] - d.box[0]) * 100}%`, height: `${(d.box[3] - d.box[1]) * 100}%` }}><span className="detection-tag">{d.label} · {Math.round(d.confidence * 100)}%</span></div>)}</div>}
                <span className="stage-label"><i />{cameraOn ? "ESP32-CAM · YOLO11n" : "CAMERA OFF"}</span>
                <span className="stage-meta">{cameraOn && result ? `${result.inference_ms} ms` : "WAITING FOR DETECTION"}</span>
              </div>
              <div className="camera-controls"><span className="control-hint">{cameraOn ? busy ? `${health?.model.name ?? "YOLO11n"} is analyzing the ESP32-CAM stream…` : "ESP32-CAM connected · automatic detection active" : "Connect to the ESP32-CAM to begin"}</span></div>
              <section className="assistant-card" aria-label="Voice assistant">
                <div className="assistant-heading"><div><p className="eyebrow">LOCAL ASSISTANT · NO API KEY</p><h3>Vision Wise</h3></div><span className={`assistant-orb ${chatBusy ? "processing" : isSpeaking ? "speaking" : voiceListening ? "listening" : ""}`} aria-hidden="true">✦</span></div>
                <p className="model-status">YOLO: {health?.model.available ? `${health.model.name} · ${health.model.class_count ?? "supported"} classes · ${health.model.device}` : health?.model.error ?? "model loading"}</p>
                <p className="model-status">{health?.assistant?.installed ? `${health.assistant.model} ready · local Ollama` : health?.assistant?.available ? `${health.assistant.model} not downloaded · run ollama pull ${health.assistant.model}` : `Ollama is not running · visual questions still work locally with YOLO ${health?.model.available ? "ready" : "unavailable"}`}</p>
                <div className="voice-controls"><button className={`wake-button ${voiceMode === "wake" ? "active" : ""}`} type="button" data-voice-toggle aria-keyshortcuts="Alt+Shift+V" onClick={toggleWakeAssistant} aria-pressed={voiceMode === "wake"} aria-label={voiceMode === "wake" ? "Turn continuous voice assistant off" : "Turn continuous voice assistant on"}>{voiceMode === "wake" ? "Turn voice assistant off" : "Activate voice assistant"}</button><button className={`voice-button ${voiceMode === "push" && voiceListening ? "listening" : ""}`} type="button" onClick={togglePushToTalk} aria-pressed={voiceMode === "push"}>{voiceMode === "push" && voiceListening ? "Stop listening" : "One question · push to talk"}</button></div>
                <p className="keyboard-access-hint">Keyboard shortcuts: <kbd>Alt</kbd> + <kbd>Shift</kbd> + <kbd>V</kbd> voice assistant · <kbd>Alt</kbd> + <kbd>Shift</kbd> + <kbd>C</kbd> camera on/off</p>
                <p className="voice-status" role="status" aria-live="polite">{voiceStatus}</p>
                <p className="privacy-disclosure">Voice assistant and camera are independent. When active, say “{settings.wake_phrase || defaultSettings.wake_phrase}” before a question. Camera start/stop commands work directly. Camera changes are logged on screen; only a cautious view-size alert is spoken automatically. Browser speech recognition may use your browser’s speech service; Vision Wise does not save audio.</p>
                <div className="assistant-detections"><strong>Current detections · left to right</strong>{result?.detections.length ? result.detections.map((item, index) => <span className="detected-word" key={`${item.label}-${index}`}>{item.label} · {item.position} · {Math.round(item.confidence * 100)}%</span>) : <span className="assistant-muted">{cameraOn ? "No recognized objects in the latest frame." : "Start the camera for a current visual description."}</span>}</div>
                {chatMessages.length === 0 && <div className="chat-suggestions" aria-label="Suggested questions"><span>Try asking</span><button type="button" disabled={chatBusy} onClick={() => void askAssistantQuestion("Describe my surroundings.")}>Describe my surroundings</button><button type="button" disabled={chatBusy} onClick={() => void askAssistantQuestion("Is there a person on my left?")}>Is anyone on my left?</button></div>}
                <form className="chat-form" onSubmit={sendAssistantMessage}><label className="visually-hidden" htmlFor="assistant-message">Ask Vision Wise</label><input id="assistant-message" value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder="Ask about your surroundings…" maxLength={1000} /><button className="chat-send" disabled={chatBusy || !chatInput.trim()}>{chatBusy ? "…" : "Ask"}</button></form>
                <div className="speech-actions"><button className="text-button" type="button" disabled={!lastResponse} onClick={repeatLastResponse}>Repeat last answer</button><button className="text-button" type="button" disabled={!isSpeaking} onClick={stopSpeaking}>Stop speaking</button></div>
                <section className="assistant-session-transcript" aria-label="Conversation and detection transcript">
                  <div className="session-transcript-heading"><strong>Live conversation & detections</strong><span>{sessionTranscript.length} updates</span></div>
                  <div className="session-transcript-list" role="log" aria-live="polite" aria-relevant="additions text">
                    {sessionTranscript.length ? sessionTranscript.map((entry) => <article className={`session-transcript-entry ${entry.kind}`} key={entry.id}><div><strong>{entry.kind === "you" ? "You said" : entry.kind === "assistant" ? "Vision Wise said" : "Camera detected"}</strong><time>{entry.time}</time></div><p>{entry.text}</p></article>) : <p className="session-transcript-empty">Turn on the assistant or camera to record questions, spoken replies, and changes in detected objects here.</p>}
                  </div>
                </section>
                {chatError && <p className="inline-warning" role="status">{chatError}</p>}
              </section>
              {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}
              {apiError && <p className="inline-error" role="status">{apiError}</p>}
              {!health?.model.available && health?.model.error && <p className="inline-warning">Model unavailable: {health.model.error}</p>}
            </article>

            <article className="panel insights-panel">
              <div className="panel-heading"><div><p className="eyebrow">SCENE NOTES</p><h2>What’s in view</h2></div><span className="result-count">{totalDetections} {totalDetections === 1 ? "object" : "objects"}</span></div>
              <div className="insight-summary"><span className="summary-icon" aria-hidden="true">✳</span><div><span className="summary-label">{result ? "LATEST FRAME" : "READY WHEN YOU ARE"}</span><p>{result?.guidance.summary ?? (cameraOn ? "Looking for objects in the camera frame…" : "Start the camera to get a concise description of the current frame.")}</p><small className="narration-status">Left and right refer to the image. This camera does not measure physical distance.</small></div></div>
              <div className="result-list" aria-label="Detected objects">
                {result?.guidance.relevant_detections.slice(0, 5).map((item, index) => <div className="object-row" key={`${item.label}-${index}`}><span className="object-index">{String(index + 1).padStart(2, "0")}</span><strong>{item.label}</strong><span className="object-position">{item.position}</span><span className="confidence"><span style={{ width: `${item.confidence * 100}%` }} /><b>{Math.round(item.confidence * 100)}%</b></span></div>)}
                {result && totalDetections === 0 && <p className="empty-results">{result.guidance.summary} Labels include estimated confidence and position in the image.</p>}
                {!result && <p className="empty-results">No results to show yet. Labels include an estimated confidence and position in the image.</p>}
              </div>
              <div className="analysis-meta"><span>Rule <b>{result?.guidance.rule ?? "—"}</b></span><span>Model <b>{result?.model.name ?? "YOLO11n"}</b></span></div>
              <div className="safety-note"><span aria-hidden="true">ⓘ</span><p>{result?.guidance.notice ?? "Object detection can be uncertain. No distance or collision prediction is provided."}</p></div>
            </article>
          </section>

          <section className="lower-grid">
            <article className="panel history-panel" id="history">
              <div className="panel-heading"><div><p className="eyebrow">RECENT ACTIVITY</p><h2>Detection history</h2></div><div><button className="text-button" onClick={() => void loadHistory()}>Refresh</button><button className="text-button clear-history" onClick={() => void clearHistory()} disabled={history.length === 0}>Clear all</button></div></div>
              <label className="search-field"><span aria-hidden="true">⌕</span><span className="visually-hidden">Search detection history</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search object summaries" /></label>
              <div className="history-list">
                {history.slice(0, 5).map((item) => <div className="history-row" key={item.id}><span className="history-marker"/><div><strong>{item.summary}</strong><span>{new Date(item.created_at).toLocaleString()} · {item.mode} mode</span></div><span className="history-time">{Math.round(item.inference_ms)} ms</span></div>)}
                {history.length === 0 && <div className="history-empty"><span aria-hidden="true">◷</span><strong>No saved detections yet</strong><p>When history is enabled, useful scene summaries will appear here.</p></div>}
              </div>
            </article>

            <article className="panel settings-panel" id="settings">
              <div className="panel-heading"><div><p className="eyebrow">MAKE IT YOURS</p><h2>Assistance settings</h2></div><span className="settings-icon">⚙</span></div>
              <label className="setting-label" htmlFor="mode">Listening mode</label>
              <select id="mode" value={settings.mode} onChange={(event) => void saveSettings({ ...settings, mode: event.target.value as Settings["mode"] })}>
                <option value="everyday">Everyday · all supported objects</option><option value="navigation">Navigation context · people and road objects</option><option value="indoor">Indoor · people and common furnishings</option>
              </select>
              <p className="setting-help">Each mode uses a visible rule set. The result is context, not a safety assessment.</p>
              <label className="setting-label" htmlFor="wake-phrase">Voice activation phrase</label>
              <input id="wake-phrase" className="setting-text" value={settings.wake_phrase} maxLength={40} onChange={(event) => setSettings({ ...settings, wake_phrase: event.target.value })} onBlur={() => void saveSettings({ ...settings, wake_phrase: settings.wake_phrase.trim() || defaultSettings.wake_phrase })} />
              <p className="setting-help">Activate the assistant, then say this phrase before each question. The default is “Vision Wise.”</p>
              <div className="slider-heading"><label htmlFor="confidence">Detection sensitivity</label><b>{Math.round(settings.confidence * 100)}% minimum</b></div>
              <input id="confidence" className="range-input" type="range" min="0.1" max="0.8" step="0.05" value={settings.confidence} onChange={(event) => void saveSettings({ ...settings, confidence: Number(event.target.value) })} />
              <p className="setting-help">A lower threshold helps catch partially visible objects. Raise it if you see false detections.</p>
              <label className="toggle-row"><span><strong>Save detection history</strong><small>Only event summaries and labels, never frames</small></span><input type="checkbox" checked={settings.save_history} onChange={(event) => void saveSettings({ ...settings, save_history: event.target.checked })} /><i aria-hidden="true" /></label>
              <label className="toggle-row speech-toggle"><span><strong>Spoken descriptions</strong><small>Uses speech already available in your browser</small></span><input type="checkbox" checked={speechOn} onChange={(event) => { speechEnabledRef.current = event.target.checked; setSpeechOn(event.target.checked); if (!event.target.checked) window.speechSynthesis?.cancel(); }} /><i aria-hidden="true" /></label>
              <label className="toggle-row"><span><strong>Continuous scene narration</strong><small>Speak when the detected scene changes</small></span><input type="checkbox" checked={settings.continuous_narration} onChange={(event) => void saveSettings({ ...settings, continuous_narration: event.target.checked })} /><i aria-hidden="true" /></label>
              <label className="setting-label voice-select-label" htmlFor="speech-voice">Speech voice</label>
              <select id="speech-voice" value={speechVoice} onChange={(event) => { setSpeechVoice(event.target.value); void saveSettings({ ...settings, speech_voice: event.target.value }); }}><option value="">System default voice</option>{availableVoices.map((voice) => <option key={`${voice.name}-${voice.lang}`} value={voice.name}>{voice.name} · {voice.lang}</option>)}</select>
              {speechOn && <div className="audio-options"><label htmlFor="speech-rate">Speech rate <b>{speechRate.toFixed(1)}×</b></label><input id="speech-rate" className="range-input" type="range" min="0.7" max="1.3" step="0.1" value={speechRate} onChange={(event) => { const value = Number(event.target.value); setSpeechRate(value); void saveSettings({ ...settings, speech_rate: value }); }} /><label htmlFor="speech-volume">Volume <b>{Math.round(speechVolume * 100)}%</b></label><input id="speech-volume" className="range-input" type="range" min="0" max="1" step="0.05" value={speechVolume} onChange={(event) => { const value = Number(event.target.value); setSpeechVolume(value); void saveSettings({ ...settings, speech_volume: value }); }} /></div>}
              <p className="setting-help">Camera: {cameraOn ? "active" : "off"}. Microphone: {voiceMode === "off" ? "off" : voiceStatus.toLowerCase()}. AI: {health?.assistant?.model ?? "checking local Ollama"}.</p>
            </article>
          </section>

          <footer className="page-footer"><span>Vision Wise <i>·</i> Local visual assistance</span><span>Not a certified mobility aid. Detections may be incorrect.</span></footer>
        </div>
      </section>
    </main>
  );
}
