"use client";

import { useVoice } from "@/hooks/use-voice";
import { Mic, Paperclip, Send, Square } from "lucide-react";
import { useRef, useState } from "react";

type Message = { role: "user" | "assistant"; content: string; error?: boolean };

/**
 * In-app chat with the CV agent.
 *
 * Talks to the project's own agent endpoint rather than Open WebUI, so replies
 * can use the classifier's tools (history, stats, model info, classification)
 * and every exchange is traced to Langfuse under one session_id.
 */
export default function AgentChat({ apiBase }: { apiBase: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [attachment, setAttachment] = useState<{
    path: string;
    filename: string;
  } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const sessionRef = useRef<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const voice = useVoice(apiBase);

  // Generated on first send, not during render: Date.now() is impure and
  // calling it while rendering violates React's purity rule.
  function sessionId(): string {
    if (!sessionRef.current) {
      sessionRef.current = `web-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    }
    return sessionRef.current;
  }

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setBusy(true);

    try {
      const res = await fetch(`${apiBase}/api/v1/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId(),
          attachment_path: attachment?.path ?? null,
        }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const data = (await res.json()) as { response: string };
      setMessages((m) => [...m, { role: "assistant", content: data.response }]);
      setAttachment(null);
      voice.speak(data.response);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: "The agent could not answer. Try again.",
          error: true,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function attach(file: File) {
    setNotice(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const res = await fetch(`${apiBase}/api/v1/chat/attachments`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(String(res.status));
      setAttachment((await res.json()) as { path: string; filename: string });
    } catch {
      setNotice("Could not attach that image; sending as text only.");
    }
  }

  async function toggleMic() {
    if (voice.isRecording) {
      const text = await voice.stopRecording();
      if (text) await send(text);
    } else {
      await voice.startRecording();
    }
  }

  return (
    <div className="agent-chat">
      <div className="messages">
        {messages.length === 0 && (
          <p className="empty">
            Ask about predictions, statistics, or the model. Attach an image to
            identify a rock.
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role} ${m.error ? "error" : ""}`}>
            {m.content}
          </div>
        ))}
        {busy && <div className="bubble assistant pending">Thinking…</div>}
      </div>

      {(voice.error || notice) && <p className="notice">{voice.error ?? notice}</p>}
      {attachment && <p className="notice">Attached: {attachment.filename}</p>}

      <div className="composer">
        <button
          onClick={() => fileRef.current?.click()}
          aria-label="Attach image"
          disabled={busy}
        >
          <Paperclip size={18} />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) attach(f);
          }}
        />
        <input
          className="prompt"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          placeholder={voice.isTranscribing ? "Transcribing…" : "Ask a question…"}
          disabled={busy}
        />
        <button
          onClick={toggleMic}
          aria-label={voice.isRecording ? "Stop recording" : "Record"}
          disabled={busy || voice.isTranscribing}
        >
          {voice.isRecording ? <Square size={18} /> : <Mic size={18} />}
        </button>
        {voice.isSpeaking && (
          <button onClick={voice.stopSpeaking} aria-label="Stop speaking">
            <Square size={18} />
          </button>
        )}
        <button onClick={() => send(input)} aria-label="Send" disabled={busy}>
          <Send size={18} />
        </button>
      </div>

      <style jsx>{`
        .agent-chat {
          display: flex;
          flex-direction: column;
          height: 100%;
          gap: 12px;
        }
        .messages {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 10px;
          padding: 8px;
        }
        .empty {
          opacity: 0.6;
          font-size: 14px;
        }
        .bubble {
          max-width: 80%;
          padding: 10px 14px;
          border-radius: 12px;
          line-height: 1.5;
          white-space: pre-wrap;
        }
        .bubble.user {
          align-self: flex-end;
          background: rgba(120, 140, 255, 0.18);
        }
        .bubble.assistant {
          align-self: flex-start;
          background: rgba(255, 255, 255, 0.06);
        }
        .bubble.error {
          background: rgba(255, 90, 90, 0.18);
        }
        .bubble.pending {
          opacity: 0.6;
        }
        .notice {
          font-size: 13px;
          opacity: 0.75;
          margin: 0 8px;
        }
        .composer {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 8px;
        }
        .composer .prompt {
          flex: 1;
          padding: 10px 12px;
          border-radius: 10px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          background: transparent;
          color: inherit;
        }
        .composer button {
          display: grid;
          place-items: center;
          width: 38px;
          height: 38px;
          border-radius: 10px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          background: transparent;
          color: inherit;
          cursor: pointer;
        }
        .composer button:disabled {
          opacity: 0.4;
          cursor: default;
        }
      `}</style>
    </div>
  );
}
