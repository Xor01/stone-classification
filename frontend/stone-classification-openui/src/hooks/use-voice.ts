"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Push-to-talk voice for the agent chat panel.
 *
 * Recording and playback both go through the backend (Whisper + OpenAI TTS)
 * rather than the browser's Web Speech API, which Firefox does not implement
 * and Safari supports only patchily.
 *
 * Speech is an enhancement: if synthesis fails the caller has already rendered
 * the text reply, so `speak` swallows its own errors rather than surfacing them.
 */
export function useVoice(apiBase: string) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.start();
      recorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setError("Microphone unavailable. You can still type your question.");
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<string> => {
    const recorder = recorderRef.current;
    if (!recorder) return "";

    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () =>
        resolve(new Blob(chunksRef.current, { type: "audio/webm" }));
      recorder.stop();
    });
    recorder.stream.getTracks().forEach((t) => t.stop());
    recorderRef.current = null;
    setIsRecording(false);

    if (blob.size === 0) return "";

    setIsTranscribing(true);
    try {
      const form = new FormData();
      form.append("audio", blob, "clip.webm");
      const res = await fetch(`${apiBase}/api/v1/voice/transcribe`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(String(res.status));
      const data = (await res.json()) as { text: string };
      return data.text;
    } catch {
      setError("Could not transcribe that. Try again or type instead.");
      return "";
    } finally {
      setIsTranscribing(false);
    }
  }, [apiBase]);

  const stopSpeaking = useCallback(() => {
    audioRef.current?.pause();
    audioRef.current = null;
    setIsSpeaking(false);
  }, []);

  const speak = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      stopSpeaking();
      try {
        const res = await fetch(`${apiBase}/api/v1/voice/speak`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error(String(res.status));
        const buffer = await res.arrayBuffer();
        const url = URL.createObjectURL(new Blob([buffer], { type: "audio/mpeg" }));
        const audio = new Audio(url);
        audioRef.current = audio;
        setIsSpeaking(true);
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setIsSpeaking(false);
        };
        await audio.play();
      } catch {
        // Speech is an enhancement; the text reply is already on screen.
        setIsSpeaking(false);
      }
    },
    [apiBase, stopSpeaking],
  );

  return {
    isRecording,
    isTranscribing,
    isSpeaking,
    error,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  };
}
