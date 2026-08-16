"use client";

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  UploadCloud,
  Image as ImageIcon,
  X,
  Loader2,
  CheckCircle2,
  XCircle,
  RefreshCw,
  BarChart3,
  History as HistoryIcon,
  ScanEye,
  AlertTriangle,
  Gauge,
  Timer,
  Layers,
  Inbox,
  MessageCircle,
} from "lucide-react";

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const MAX_FILE_MB = 15;

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

type TabId = "classify" | "history" | "stats" | "chat";
type HealthState = "checking" | "online" | "offline";

interface TopKEntry {
  class_name: string;
  probability: number; // 0..1
}

interface PredictionResult {
  predicted_class: string;
  confidence: number; // 0..1
  inference_ms: number;
  top_predictions?: TopKEntry[];
}

interface HistoryItem {
  id: string | number;
  image_name: string;
  predicted_class: string;
  confidence: number; // 0..1
  inference_ms: number;
  created_at: string;
}

interface StatsData {
  total_predictions: number;
  average_confidence: number; // 0..1
  class_distribution: Record<string, number>;
}

interface ToastMessage {
  id: number;
  kind: "error" | "info";
  text: string;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const pct = (n: number) => `${Math.round(n * 100)}%`;

const formatTimestamp = (iso: string) => {
  try {
    const d = new Date(iso);
    return d.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Riyadh",
    });
  } catch {
    return iso;
  }
};

const classAccentPalette = [
  "#3F5F7A", // cyanotype steel
  "#C1873F", // raincoat ochre
  "#7C8A6E", // fog moss
  "#A8562F", // rock rust
  "#5C6B6F", // rock grey-blue
  "#8A6A3F", // raincoat brown
];

const accentForClass = (name: string) => {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return classAccentPalette[h % classAccentPalette.length];
};

// ----------------------------------------------------------------------------
// Root Page
// ----------------------------------------------------------------------------

export default function Page() {
  const [activeTab, setActiveTab] = useState<TabId>("classify");
  const [health, setHealth] = useState<HealthState>("checking");
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const pushToast = useCallback((text: string, kind: ToastMessage["kind"] = "error") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, kind, text }]);
    window.setTimeout(() => {
      setToasts((t) => t.filter((m) => m.id !== id));
    }, 5200);
  }, []);

  const dismissToast = (id: number) => setToasts((t) => t.filter((m) => m.id !== id));

  // Poll backend health
  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        if (!cancelled) setHealth(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setHealth("offline");
      }
    };

    check();
    const interval = window.setInterval(check, 20000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <main className="page">
      <SpecimenHero health={health} />

      <Header activeTab={activeTab} onTabChange={setActiveTab} health={health} />

      <div className="content">
        <div className="view-flow" key={activeTab}>
          {activeTab === "classify" && (
            <ClassifyView health={health} onError={(m) => pushToast(m, "error")} />
          )}
          {activeTab === "history" && <HistoryView onError={(m) => pushToast(m, "error")} />}
          {activeTab === "stats" && <StatsView onError={(m) => pushToast(m, "error")} />}
          {activeTab === "chat" && <ChatView />}
        </div>
      </div>

      <ToastStack toasts={toasts} onDismiss={dismissToast} />

      <style jsx global>{`
        :root {
          /* Fogged paper — page ground, between the rock photo's grey air and raincoat cream */
          --fog: #eef0e7;
          --mist: #d8d3c1;
          --slate: #6a6a5c;
          --slate-dark: #45463a;
          --charcoal: #201f1a;
          /* Cyanotype indigo — the plate the specimen sits on */
          --cyanotype: #1e3a5c;
          --cyanotype-deep: #142943;
          --cyanotype-pale: #a9c1d6;
          --accent: #1e3a5c;
          --accent-warm: #c1873f;
          --success: #5c9a6c;
          --error: #b2543f;
          --error-text: #8c4030;
          --line: rgba(32, 31, 26, 0.12);
          --card: #faf7ed;
          --focus-ring: rgba(30, 58, 92, 0.35);
        }

        *,
        *::before,
        *::after {
          box-sizing: border-box;
        }

        html {
          background: var(--fog);
        }

        html,
        body {
          margin: 0;
          padding: 0;
        }

        body {
          font-family: var(--font-body);
          color: var(--charcoal);
          line-height: 1.5;
          -webkit-font-smoothing: antialiased;
          text-rendering: optimizeLegibility;
        }

        h1,
        h2,
        h3,
        p {
          margin: 0;
        }

        button {
          font-family: inherit;
          cursor: pointer;
        }

        button:disabled {
          cursor: not-allowed;
        }

        img {
          max-width: 100%;
        }

        /* Consistent, visible keyboard focus across every interactive element */
        a:focus-visible,
        button:focus-visible,
        input:focus-visible,
        [role="button"]:focus-visible,
        [tabindex]:focus-visible {
          outline: 2px solid var(--accent);
          outline-offset: 2px;
          box-shadow: 0 0 0 5px var(--focus-ring);
          border-radius: 6px;
        }

        @media (prefers-reduced-motion: reduce) {
          *,
          *::before,
          *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
          }
        }
      `}</style>

      <style jsx>{`
        .page {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }
        .content {
          flex: 1;
          width: 100%;
          max-width: 1080px;
          margin: 0 auto;
          padding: 32px 24px 96px;
          overflow: clip;
        }
        .view-flow {
          animation: viewIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes viewIn {
          from {
            opacity: 0;
            transform: translateY(10px);
            filter: blur(2px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
          }
        }
        @media (max-width: 640px) {
          .content {
            padding: 20px 16px 80px;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .view-flow {
            animation: none;
          }
        }
      `}</style>
    </main>
  );
}

// ----------------------------------------------------------------------------
// Specimen Hero — a cyanotype "plate" of the rock, with a paper field-tag
// pinned over it. The plate borrows the exposure-blue of a cyanotype print;
// the tag borrows the soft, hand-inked warmth of the raincoat illustration;
// the rock is a rough, textured mass sitting low in real, moving water — the
// water drifts on its own timing and disturbs itself faintly where you rest
// or drag the pointer, rather than snapping the whole scene toward it.
// ----------------------------------------------------------------------------

const WATER_TOP = 232; // viewBox y where the rock meets the surface
const HERO_VB_W = 1200;
const HERO_VB_H = 340;

function SpecimenHero({ health }: { health: HealthState }) {
  const svgRef = useRef<SVGSVGElement>(null);

  return (
    <div className="hero" role="presentation" aria-hidden="true">
      <svg
        ref={svgRef}
        className="hero-svg"
        viewBox={`0 0 ${HERO_VB_W} ${HERO_VB_H}`}
        preserveAspectRatio="xMidYMax slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="plate" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#24466e" />
            <stop offset="100%" stopColor="#152944" />
          </linearGradient>
          <radialGradient id="expose" cx="62%" cy="14%" r="75%">
            <stop offset="0%" stopColor="#3f6791" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#3f6791" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="waterline" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1c3654" />
            <stop offset="100%" stopColor="#0e2036" />
          </linearGradient>
          <clipPath id="waterClip">
            <rect x="0" y={WATER_TOP} width={HERO_VB_W} height={HERO_VB_H - WATER_TOP} />
          </clipPath>
          <clipPath id="rockClip">
            <use href="#rock-main" />
            <use href="#rock-small" />
          </clipPath>

          {/* rough mineral grain, clipped onto the rock silhouette only */}
          <filter id="rockGrain" x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence type="fractalNoise" baseFrequency="0.14 0.16" numOctaves="3" seed="7" result="n" />
            <feColorMatrix
              in="n"
              type="matrix"
              values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0.5 0.5 0.5 0 0"
            />
          </filter>

          {/* faceted rock — soft interior sheen so each plane reads as polished mineral, not flat paint */}
          <linearGradient id="facetSheenA" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          <radialGradient id="rockGlint" cx="30%" cy="20%" r="60%">
            <stop offset="0%" stopColor="#eaf3ff" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#eaf3ff" stopOpacity="0" />
          </radialGradient>
        </defs>

        <rect x="0" y="0" width={HERO_VB_W} height={HERO_VB_H} fill="url(#plate)" />
        <rect x="0" y="0" width={HERO_VB_W} height={HERO_VB_H} fill="url(#expose)" />

        {/* the water — surface + drifting waves + reflection + disturbance, clipped to the water band */}
        <g clipPath="url(#waterClip)">
          <rect x="0" y={WATER_TOP} width={HERO_VB_W} height={HERO_VB_H - WATER_TOP} fill="url(#waterline)" />

          {/* rock's reflection, flipped and softened */}
          <g className="reflection" transform={`translate(0 ${WATER_TOP * 2}) scale(1 -1)`} opacity="0.3">
            <use href="#rock-main" />
            <use href="#rock-small" />
          </g>

          <g className="wave-a">
            <path
              d="M0 20 Q 50 8 100 20 T 200 20 T 300 20 T 400 20 T 500 20 T 600 20 T 700 20 T 800 20 T 900 20 T 1000 20 T 1100 20 T 1200 20 V120 H0 Z"
              transform={`translate(0 ${WATER_TOP - 8})`}
              fill="#25476c"
              opacity="0.8"
            />
          </g>
          <g transform={`translate(0 ${WATER_TOP + 14})`}>
            <g className="wave-b">
              <path
                d="M0 16 Q 70 26 140 16 T 280 16 T 420 16 T 560 16 T 700 16 T 840 16 T 980 16 T 1120 16 T 1260 16 V120 H0 Z"
                fill="#193453"
                opacity="0.85"
              />
            </g>
          </g>
          <g transform={`translate(0 ${WATER_TOP + 40})`}>
            <g className="wave-c">
              <path
                d="M0 12 Q 40 20 80 12 T 160 12 T 240 12 T 320 12 T 400 12 T 480 12 T 560 12 T 640 12 T 720 12 T 800 12 T 880 12 T 960 12 T 1040 12 T 1120 12 T 1200 12 V120 H0 Z"
                fill="#102337"
                opacity="0.9"
              />
            </g>
          </g>

          {/* fine, ever-drifting surface glints */}
          <g className="glints" stroke="#8fb0cf" strokeWidth="1.4" opacity="0.4" strokeLinecap="round">
            <path d="M 90 254 Q 140 250 190 254" />
            <path d="M 320 268 Q 380 262 440 268" />
            <path d="M 560 250 Q 610 246 660 250" />
            <path d="M 760 274 Q 820 268 880 274" />
            <path d="M 950 258 Q 1000 253 1050 258" />
          </g>
        </g>

        {/* the rock — rough, mineral, legible against the plate */}
        <g className="rock-settle">
          <ellipse cx="660" cy={WATER_TOP + 2} rx="270" ry="9" fill="#08151f" opacity="0.4" />

          {/* small companion rock, partly behind, as in the reference photo */}
          <path
            id="rock-small"
            d="M 792 232 L 800 216 L 811 208 L 826 205 L 840 210 L 852 220 L 862 232 Z"
            fill="#17324f"
          />
          <g stroke="#0b1e33" strokeWidth="1.1" opacity="0.6" strokeLinecap="round">
            <path d="M 811 208 L 816 232" />
            <path d="M 833 207 L 828 232" />
          </g>

          {/* main rock: jagged outer silhouette */}
          <path
            id="rock-main"
            d="M 398 232
               L 406 210 L 418 198 L 424 208 L 438 182 L 452 190 L 462 168
               L 478 176 L 496 154 L 512 162 L 528 138 L 546 148 L 566 128
               L 584 126 L 602 118 L 618 128 L 636 122 L 654 130 L 672 118
               L 688 128 L 704 116 L 720 130 L 736 124 L 750 140 L 764 148
               L 778 168 L 792 174 L 804 194 L 818 200 L 828 218 L 838 222
               L 846 232 Z"
            fill="#233f60"
          />
          {/* facets — five planes instead of a smooth gradient, for a rough-cut read */}
          <path
            d="M 424 208 L 438 182 L 452 190 L 462 168 L 478 176 L 496 154 L 512 162
               L 528 138 L 546 148 L 566 128 L 584 126
               L 588 172 L 552 186 L 512 196 L 470 210 L 438 220 Z"
            fill="#5b83ac"
          />
          <path
            d="M 584 126 L 602 118 L 618 128 L 636 122 L 654 130 L 672 118 L 688 128 L 704 116
               L 706 158 L 664 168 L 620 172 L 588 172 Z"
            fill="#3f6892"
          />
          <path d="M 398 232 L 406 210 L 418 198 L 424 208 L 438 220 L 470 210 L 448 232 Z" fill="#2c5178" />
          <path
            d="M 704 116 L 720 130 L 736 124 L 750 140 L 764 148 L 778 168 L 792 174
               L 794 214 L 748 208 L 706 190 L 706 158 Z"
            fill="#1c3c5e"
          />
          <path
            d="M 792 174 L 804 194 L 818 200 L 828 218 L 838 222 L 846 232
               L 794 232 L 794 214 Z"
            fill="#122843"
          />

          {/* rough mineral grain across the whole rock */}
          <g clipPath="url(#rockClip)" opacity="0.5" style={{ mixBlendMode: "overlay" }}>
            <rect x="380" y="100" width="500" height="140" filter="url(#rockGrain)" />
          </g>

          {/* fracture lines — uneven, some branching, like a real rough-cut face */}
          <g stroke="#0d2038" strokeWidth="1.3" opacity="0.65" strokeLinecap="round" fill="none">
            <path d="M 470 180 L 486 210 L 480 232" />
            <path d="M 486 210 L 504 218" />
            <path d="M 600 150 L 610 188 L 596 224" />
            <path d="M 660 150 L 672 176" />
            <path d="M 720 168 L 738 196 L 732 222" />
            <path d="M 460 198 L 474 202" />
          </g>
          {/* crease highlights along the facet edges */}
          <g stroke="#eaf1f6" strokeWidth="1.3" opacity="0.5" strokeLinecap="round" fill="none">
            <path d="M 584 126 L 588 172 L 620 172 L 664 168 L 706 158" />
            <path d="M 704 116 L 706 158 L 706 190 L 748 208 L 794 214" />
            <path d="M 424 208 L 438 220" />
          </g>
          {/* pitting — small pocked hollows, dark and light pairs for depth */}
          <g opacity="0.55">
            <ellipse cx="512" cy="176" rx="5" ry="3" fill="#0d2038" />
            <ellipse cx="514" cy="174" rx="2" ry="1.2" fill="#7ea3c6" opacity="0.6" />
            <ellipse cx="650" cy="146" rx="4" ry="2.6" fill="#0d2038" />
            <ellipse cx="740" cy="182" rx="4.5" ry="3" fill="#0a1a2c" />
            <ellipse cx="742" cy="180" rx="2" ry="1.1" fill="#7ea3c6" opacity="0.5" />
          </g>
          {/* mineral flecks catching the light */}
          <g fill="#dfeaf3" opacity="0.75">
            <circle cx="560" cy="158" r="1.4" />
            <circle cx="628" cy="140" r="1.6" />
            <circle cx="680" cy="150" r="1.2" />
            <circle cx="546" cy="192" r="1.3" />
            <circle cx="760" cy="172" r="1.3" />
          </g>

          {/* glassy sheen across the crown facets — the polish that makes it read as a gem-cut mineral, not flat paper */}
          <g clipPath="url(#rockClip)">
            <path
              d="M 424 208 L 438 182 L 452 190 L 462 168 L 478 176 L 496 154 L 512 162
                 L 528 138 L 546 148 L 566 128 L 584 126 L 588 172 L 552 186 L 512 196 L 470 210 L 438 220 Z"
              fill="url(#facetSheenA)"
            />
            <ellipse cx="540" cy="150" rx="220" ry="70" fill="url(#rockGlint)" />
          </g>

          {/* crisp rim light around the jagged silhouette for clarity against the plate */}
          <path
            d="M 398 232
               L 406 210 L 418 198 L 424 208 L 438 182 L 452 190 L 462 168
               L 478 176 L 496 154 L 512 162 L 528 138 L 546 148 L 566 128
               L 584 126 L 602 118 L 618 128 L 636 122 L 654 130 L 672 118
               L 688 128 L 704 116 L 720 130 L 736 124 L 750 140 L 764 148
               L 778 168 L 792 174 L 804 194 L 818 200 L 828 218 L 838 222 L 846 232"
            fill="none"
            stroke="#eef3f8"
            strokeWidth="1.8"
            strokeLinejoin="round"
            strokeLinecap="round"
            opacity="0.85"
          />
          <path
            d="M 792 232 L 800 216 L 811 208 L 826 205 L 840 210 L 852 220 L 862 232"
            fill="none"
            stroke="#dbe7f0"
            strokeWidth="1.4"
            opacity="0.65"
          />
        </g>
      </svg>

      {/* the field tag — paper, deckled, pinned over the plate */}
      <div className="tag">
        <span className={`stamp ${health}`}>
          <span className="dot" />
          {health === "online" ? "Agent online" : health === "offline" ? "Agent offline" : "Checking agent…"}
        </span>
        <h1>Specimen Classification</h1>
        <p>Field identification for geological samples</p>
      </div>

      <style jsx>{`
        .hero {
          position: relative;
          height: 248px;
          width: 100%;
          overflow: hidden;
          background: #152944;
        }
        .hero-svg {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          display: block;
        }
        .rock-settle {
          animation: settle 9s ease-in-out infinite;
          transform-origin: 620px 232px;
        }
        @keyframes settle {
          0%,
          100% {
            transform: translateY(0);
          }
          45% {
            transform: translateY(-1.5px);
          }
          70% {
            transform: translateY(-0.5px);
          }
        }
        .wave-a path {
          animation: driftA 8s ease-in-out infinite;
        }
        .wave-b {
          animation: driftB 11s ease-in-out infinite;
        }
        .wave-c {
          animation: driftC 15s ease-in-out infinite;
        }
        @keyframes driftA {
          0%,
          100% {
            d: path(
              "M0 20 Q 50 8 100 20 T 200 20 T 300 20 T 400 20 T 500 20 T 600 20 T 700 20 T 800 20 T 900 20 T 1000 20 T 1100 20 T 1200 20 V120 H0 Z"
            );
          }
          38% {
            d: path(
              "M0 15 Q 50 25 100 15 T 200 15 T 300 15 T 400 15 T 500 15 T 600 15 T 700 15 T 800 15 T 900 15 T 1000 15 T 1100 15 T 1200 15 V120 H0 Z"
            );
          }
          72% {
            d: path(
              "M0 24 Q 50 12 100 24 T 200 24 T 300 24 T 400 24 T 500 24 T 600 24 T 700 24 T 800 24 T 900 24 T 1000 24 T 1100 24 T 1200 24 V120 H0 Z"
            );
          }
        }
        @keyframes driftB {
          0%,
          100% {
            transform: translateX(0);
          }
          40% {
            transform: translateX(-22px);
          }
          75% {
            transform: translateX(10px);
          }
        }
        @keyframes driftC {
          0%,
          100% {
            transform: translateX(0);
          }
          35% {
            transform: translateX(16px);
          }
          68% {
            transform: translateX(-12px);
          }
        }
        .glints {
          animation: twinkle 5s ease-in-out infinite;
        }
        @keyframes twinkle {
          0%,
          100% {
            opacity: 0.22;
          }
          50% {
            opacity: 0.48;
          }
        }
        .reflection {
          animation: shimmerRefl 6s ease-in-out infinite;
        }
        @keyframes shimmerRefl {
          0%,
          100% {
            opacity: 0.24;
          }
          50% {
            opacity: 0.36;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .rock-settle,
          .wave-a path,
          .wave-b,
          .wave-c,
          .glints,
          .reflection {
            animation: none !important;
          }
        }
        .tag {
          position: absolute;
          left: 50%;
          bottom: 22px;
          transform: translateX(-50%) rotate(-0.6deg);
          text-align: center;
          width: min(420px, calc(100% - 48px));
          background: var(--card);
          color: var(--charcoal);
          padding: 20px 24px 22px;
          border-radius: 3px;
          box-shadow: 0 14px 30px rgba(10, 20, 34, 0.35);
          border: 1px solid rgba(32, 31, 26, 0.14);
          pointer-events: none;
        }
        .tag::before {
          content: "";
          position: absolute;
          inset: 6px;
          border: 1px dashed rgba(32, 31, 26, 0.22);
          border-radius: 2px;
          pointer-events: none;
        }
        .stamp {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-family: var(--font-mono);
          font-size: 10.5px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--slate-dark);
          margin-bottom: 10px;
        }
        .dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--mist);
        }
        .stamp.online .dot {
          background: var(--success);
          box-shadow: 0 0 0 3px rgba(92, 154, 108, 0.22);
        }
        .stamp.offline .dot {
          background: var(--error);
          box-shadow: 0 0 0 3px rgba(178, 84, 63, 0.22);
        }
        h1 {
          font-family: var(--font-display);
          font-weight: 500;
          font-style: italic;
          font-size: clamp(24px, 3.6vw, 32px);
          letter-spacing: -0.01em;
          color: var(--charcoal);
        }
        p {
          margin-top: 5px;
          font-size: 13px;
          color: var(--slate);
          letter-spacing: 0.01em;
        }
        @media (max-width: 640px) {
          .hero {
            height: 216px;
          }
          .tag {
            padding: 16px 18px 18px;
          }
        }
      `}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// Header — sticky glass bar with tabs + health badge
// ----------------------------------------------------------------------------

function Header({
  activeTab,
  onTabChange,
  health,
}: {
  activeTab: TabId;
  onTabChange: (t: TabId) => void;
  health: HealthState;
}) {
  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: "classify", label: "Classify", icon: <ScanEye size={16} strokeWidth={2} /> },
    { id: "history", label: "Prediction History", icon: <HistoryIcon size={16} strokeWidth={2} /> },
    { id: "stats", label: "Operational Stats", icon: <BarChart3 size={16} strokeWidth={2} /> },
    { id: "chat", label: "Chat", icon: <MessageCircle size={16} strokeWidth={2} /> },
  ];

  const tabRefs = useRef<Partial<Record<TabId, HTMLButtonElement | null>>>({});
  const [indicator, setIndicator] = useState({ left: 0, width: 0, ready: false });

  const measure = useCallback(() => {
    const el = tabRefs.current[activeTab];
    if (el) setIndicator({ left: el.offsetLeft, width: el.offsetWidth, ready: true });
  }, [activeTab]);

  useEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  return (
    <header className="header">
      <div className="header-inner">
        <nav className="tabs" aria-label="Views">
          <span
            className="tab-indicator"
            style={{
              transform: `translateX(${indicator.left}px)`,
              width: indicator.width,
              opacity: indicator.ready ? 1 : 0,
            }}
          />
          {tabs.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el;
              }}
              className={`tab ${activeTab === t.id ? "active" : ""}`}
              onClick={() => onTabChange(t.id)}
              aria-current={activeTab === t.id ? "page" : undefined}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </nav>

        <div className={`health-badge ${health}`} role="status">
          <span className="health-dot" />
          {health === "checking" ? "Checking" : health === "online" ? "Online" : "Offline"}
        </div>
      </div>

      <style jsx>{`
        .header {
          position: sticky;
          top: 0;
          z-index: 30;
          background: rgba(238, 240, 231, 0.86);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border-bottom: 1px solid var(--line);
        }
        .header-inner {
          max-width: 1080px;
          margin: 0 auto;
          padding: 12px 24px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
        }
        .tabs {
          position: relative;
          display: flex;
          gap: 4px;
          background: rgba(28, 33, 40, 0.05);
          padding: 4px;
          border-radius: 12px;
          overflow-x: auto;
        }
        .tab-indicator {
          position: absolute;
          top: 4px;
          bottom: 4px;
          left: 0;
          border-radius: 9px;
          background: linear-gradient(135deg, var(--cyanotype) 0%, var(--cyanotype-deep) 100%);
          box-shadow: 0 4px 14px rgba(30, 58, 92, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.12);
          transition: transform 0.55s cubic-bezier(0.16, 1, 0.3, 1), width 0.55s cubic-bezier(0.16, 1, 0.3, 1),
            opacity 0.3s ease;
          pointer-events: none;
          z-index: 0;
        }
        .tab {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          gap: 6px;
          border: none;
          background: transparent;
          padding: 8px 14px;
          border-radius: 9px;
          font-family: var(--font-body);
          font-size: 13.5px;
          font-weight: 500;
          color: var(--slate);
          white-space: nowrap;
          transition: color 0.35s ease, transform 0.2s ease;
        }
        .tab:hover {
          color: var(--charcoal);
        }
        .tab:active {
          transform: scale(0.97);
        }
        .tab.active {
          color: #f4f2e8;
        }
        .health-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          font-family: var(--font-mono);
          font-size: 12px;
          font-weight: 500;
          padding: 6px 12px;
          border-radius: 100px;
          border: 1px solid var(--line);
          color: var(--slate);
          flex-shrink: 0;
        }
        .health-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #9aa5a8;
        }
        .health-badge.online {
          color: #2f7a54;
          border-color: rgba(95, 185, 138, 0.35);
          background: rgba(95, 185, 138, 0.08);
        }
        .health-badge.online .health-dot {
          background: var(--success);
          box-shadow: 0 0 0 3px rgba(95, 185, 138, 0.22);
          animation: pulse 2s ease-in-out infinite;
        }
        .health-badge.offline {
          color: #a13d31;
          border-color: rgba(193, 87, 74, 0.3);
          background: rgba(193, 87, 74, 0.07);
        }
        .health-badge.offline .health-dot {
          background: var(--error);
        }
        @keyframes pulse {
          0%,
          100% {
            box-shadow: 0 0 0 3px rgba(95, 185, 138, 0.22);
          }
          50% {
            box-shadow: 0 0 0 6px rgba(95, 185, 138, 0.1);
          }
        }
        @media (max-width: 640px) {
          .header-inner {
            padding: 10px 16px;
            flex-wrap: wrap;
          }
          .tab span {
            display: none;
          }
          .tab {
            padding: 8px 10px;
          }
        }
      `}</style>
    </header>
  );
}

// ----------------------------------------------------------------------------
// Toasts
// ----------------------------------------------------------------------------

function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className="stack" role="alert" aria-live="assertive">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <AlertTriangle size={16} strokeWidth={2} />
          <span>{t.text}</span>
          <button aria-label="Dismiss" onClick={() => onDismiss(t.id)}>
            <X size={14} strokeWidth={2} />
          </button>
        </div>
      ))}
      <style jsx>{`
        .stack {
          position: fixed;
          right: 20px;
          bottom: 20px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          z-index: 50;
          max-width: min(360px, calc(100vw - 40px));
        }
        .toast {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          background: var(--charcoal);
          color: #fff;
          padding: 12px 12px 12px 14px;
          border-radius: 10px;
          font-size: 13px;
          line-height: 1.4;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
          animation: rise 0.24s ease;
        }
        .toast.error {
          border-left: 3px solid var(--error);
        }
        .toast span {
          flex: 1;
        }
        .toast button {
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.6);
          flex-shrink: 0;
          padding: 2px;
          border-radius: 4px;
        }
        .toast button:hover {
          color: #fff;
        }
        @keyframes rise {
          from {
            opacity: 0;
            transform: translateY(6px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// View 1 — Classify
// ----------------------------------------------------------------------------

function ClassifyView({
  health,
  onError,
}: {
  health: HealthState;
  onError: (msg: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [barsFilled, setBarsFilled] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const validateAndSetFile = useCallback(
    (f: File) => {
      if (!ACCEPTED_TYPES.includes(f.type)) {
        onError("Unsupported format. Please upload a JPG, PNG, or WEBP image.");
        return;
      }
      if (f.size > MAX_FILE_MB * 1024 * 1024) {
        onError(`File too large. Keep uploads under ${MAX_FILE_MB}MB.`);
        return;
      }
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setFile(f);
      setPreviewUrl(URL.createObjectURL(f));
      setResult(null);
    },
    [previewUrl, onError]
  );

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) validateAndSetFile(f);
  };

  const onBrowse = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) validateAndSetFile(f);
    e.target.value = "";
  };

  const removeFile = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  const runPrediction = async () => {
    if (!file) return;
    if (health === "offline") {
      onError("Backend is unreachable. Prediction cannot be run right now.");
      return;
    }
    setLoading(true);
    setResult(null);
    setBarsFilled(false);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(`${API_BASE}/api/v1/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Prediction failed (${res.status})`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      window.setTimeout(() => setBarsFilled(true), 50);
    } catch (err) {
      onError(
        err instanceof Error
          ? `Could not run prediction: ${err.message}`
          : "Could not run prediction. The backend may be unreachable."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="classify" aria-label="Image classification">
      <div className="grid">
        <div className="card upload-card">
          <h2>Upload specimen</h2>
          <p className="hint">JPG, PNG, or WEBP · up to {MAX_FILE_MB}MB</p>

          {!previewUrl ? (
            <div
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload an image by dragging it here or pressing enter to browse"
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
              }}
            >
              <UploadCloud size={30} strokeWidth={1.6} />
              <p className="drop-title">Drag and drop an image</p>
              <p className="drop-sub">or click to browse your files</p>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                hidden
                onChange={onBrowse}
              />
            </div>
          ) : (
            <div className="preview">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewUrl} alt="Selected specimen preview" />
              <div className="preview-actions">
                <button className="ghost" onClick={() => inputRef.current?.click()}>
                  Replace
                </button>
                <button className="ghost danger" onClick={removeFile}>
                  <X size={14} strokeWidth={2} /> Remove
                </button>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                hidden
                onChange={onBrowse}
              />
            </div>
          )}

          <button
            className="cta"
            disabled={!file || loading || health === "offline"}
            onClick={runPrediction}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spin" strokeWidth={2} /> Running prediction…
              </>
            ) : (
              <>
                <ScanEye size={16} strokeWidth={2} /> Upload / Predict
              </>
            )}
          </button>
          {health === "offline" && (
            <p className="offline-note">
              <XCircle size={13} strokeWidth={2} /> Backend offline — reconnect to classify.
            </p>
          )}
        </div>

        <div className="card result-card">
          <h2>Result</h2>

          {loading && <ResultSkeleton />}

          {!loading && !result && (
            <div className="empty">
              <ImageIcon size={26} strokeWidth={1.5} />
              <p>Upload an image and run a prediction to see results here.</p>
            </div>
          )}

          {!loading && result && (
            <div className="result-body">
              <div className="predicted">
                <span className="label">Predicted class</span>
                <span className="value">{result.predicted_class}</span>
              </div>

              <div className="metric-row">
                <div className="metric">
                  <div className="metric-head">
                    <Gauge size={14} strokeWidth={2} />
                    <span>Confidence</span>
                    <span className="metric-num">{pct(result.confidence)}</span>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: barsFilled ? pct(result.confidence) : "0%",
                        background: accentForClass(result.predicted_class),
                      }}
                    />
                  </div>
                </div>

                <div className="latency-chip">
                  <Timer size={14} strokeWidth={2} />
                  <span>{Math.round(result.inference_ms)} ms</span>
                </div>
              </div>

              {result.top_predictions && result.top_predictions.length > 0 && (
                <div className="topk">
                  <div className="topk-head">
                    <Layers size={13} strokeWidth={2} />
                    <span>Alternative predictions</span>
                  </div>
                  {result.top_predictions.map((entry, i) => (
                    <div className="topk-row" key={`${entry.class_name}-${i}`}>
                      <span className="topk-name">{entry.class_name}</span>
                      <div className="topk-track">
                        <div
                          className="topk-fill"
                          style={{
                            width: barsFilled ? pct(entry.probability) : "0%",
                            background: accentForClass(entry.class_name),
                          }}
                        />
                      </div>
                      <span className="topk-num">{pct(entry.probability)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          align-items: start;
        }
        @media (max-width: 860px) {
          .grid {
            grid-template-columns: 1fr;
          }
        }
        .card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 24px;
        }
        h2 {
          font-family: var(--font-display);
          font-weight: 500;
          font-size: 19px;
          margin-bottom: 4px;
        }
        .hint {
          font-size: 12.5px;
          color: var(--slate);
          margin-bottom: 18px;
        }
        .dropzone {
          border: 1.5px dashed var(--mist);
          border-radius: 16px;
          padding: 40px 20px;
          text-align: center;
          color: var(--slate);
          cursor: pointer;
          transition: border-color 0.3s ease, background 0.3s ease, transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .dropzone:hover,
        .dropzone.dragging {
          border-color: var(--accent);
          background: rgba(30, 58, 92, 0.05);
          color: var(--charcoal);
          transform: translateY(-2px);
        }
        .dropzone.dragging {
          transform: scale(1.01);
        }
        .drop-title {
          font-size: 14.5px;
          font-weight: 600;
          margin-top: 10px;
        }
        .drop-sub {
          font-size: 12.5px;
          margin-top: 2px;
          color: var(--slate);
        }
        .preview {
          position: relative;
          border-radius: 14px;
          overflow: hidden;
          border: 1px solid var(--line);
        }
        .preview img {
          width: 100%;
          height: 230px;
          object-fit: cover;
          display: block;
        }
        .preview-actions {
          position: absolute;
          top: 10px;
          right: 10px;
          display: flex;
          gap: 6px;
        }
        .ghost {
          display: flex;
          align-items: center;
          gap: 4px;
          background: rgba(28, 33, 40, 0.65);
          backdrop-filter: blur(4px);
          color: #fff;
          border: none;
          font-size: 12px;
          padding: 6px 10px;
          border-radius: 100px;
          transition: background 0.25s ease, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .ghost:hover {
          background: rgba(28, 33, 40, 0.85);
          transform: translateY(-1px);
        }
        .ghost.danger:hover {
          background: var(--error);
        }
        .cta {
          position: relative;
          overflow: hidden;
          width: 100%;
          margin-top: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: linear-gradient(135deg, #2a4d76 0%, var(--cyanotype) 45%, var(--cyanotype-deep) 100%);
          color: #f4f2e8;
          border: none;
          padding: 14px;
          border-radius: 12px;
          font-size: 14px;
          font-weight: 600;
          letter-spacing: 0.01em;
          box-shadow: 0 6px 18px rgba(20, 41, 67, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.14);
          transition: box-shadow 0.3s ease, transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .cta::before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(
            110deg,
            transparent 20%,
            rgba(255, 255, 255, 0.22) 40%,
            transparent 60%
          );
          transform: translateX(-120%);
          transition: transform 0.7s ease;
        }
        .cta:hover:not(:disabled)::before {
          transform: translateX(120%);
        }
        .cta:hover:not(:disabled) {
          box-shadow: 0 10px 26px rgba(20, 41, 67, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.18);
          transform: translateY(-1px);
        }
        .cta:active:not(:disabled) {
          transform: translateY(0) scale(0.98);
        }
        .cta:disabled {
          background: var(--mist);
          color: #837e6b;
          box-shadow: none;
          cursor: not-allowed;
        }
        .spin {
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
        .offline-note {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: var(--error-text);
          margin-top: 10px;
        }
        .empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          gap: 10px;
          color: var(--slate);
          padding: 56px 12px;
          font-size: 13.5px;
        }
        .result-body {
          animation: fadeIn 0.25s ease;
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .predicted {
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding-bottom: 16px;
          border-bottom: 1px solid var(--line);
          margin-bottom: 16px;
        }
        .predicted .label {
          font-size: 11.5px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--slate);
          font-family: var(--font-mono);
        }
        .predicted .value {
          font-family: var(--font-display);
          font-size: 28px;
          font-weight: 600;
          color: var(--charcoal);
        }
        .metric-row {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 18px;
        }
        .metric {
          flex: 1;
        }
        .metric-head {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12.5px;
          color: var(--slate);
          margin-bottom: 6px;
        }
        .metric-num {
          margin-left: auto;
          font-family: var(--font-mono);
          font-weight: 600;
          color: var(--charcoal);
        }
        .bar-track {
          height: 8px;
          border-radius: 100px;
          background: rgba(28, 33, 40, 0.07);
          overflow: hidden;
        }
        .bar-fill {
          height: 100%;
          border-radius: 100px;
          transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .latency-chip {
          display: flex;
          align-items: center;
          gap: 6px;
          font-family: var(--font-mono);
          font-size: 12.5px;
          font-weight: 500;
          color: var(--charcoal);
          background: rgba(28, 33, 40, 0.05);
          padding: 8px 12px;
          border-radius: 10px;
          flex-shrink: 0;
        }
        .topk {
          padding-top: 4px;
        }
        .topk-head {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11.5px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--slate);
          font-family: var(--font-mono);
          margin-bottom: 10px;
        }
        .topk-row {
          display: grid;
          grid-template-columns: 92px 1fr 44px;
          align-items: center;
          gap: 10px;
          margin-bottom: 9px;
        }
        .topk-name {
          font-size: 12.5px;
          color: var(--charcoal);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .topk-track {
          height: 6px;
          border-radius: 100px;
          background: rgba(28, 33, 40, 0.06);
          overflow: hidden;
        }
        .topk-fill {
          height: 100%;
          border-radius: 100px;
          transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1);
          opacity: 0.8;
        }
        .topk-num {
          font-family: var(--font-mono);
          font-size: 11.5px;
          text-align: right;
          color: var(--slate);
        }
      `}</style>
    </section>
  );
}

function ResultSkeleton() {
  return (
    <div className="skel">
      <div className="s-line w60" />
      <div className="s-line w40 h28" />
      <div className="s-line w100 h8" style={{ marginTop: 18 }} />
      <div className="s-line w100 h8" />
      <div className="s-line w100 h8" />
      <style jsx>{`
        .skel {
          padding-top: 4px;
        }
        .s-line {
          background: linear-gradient(90deg, #eceff0 0%, #f6f7f7 50%, #eceff0 100%);
          background-size: 200% 100%;
          animation: shimmer 1.4s ease-in-out infinite;
          border-radius: 6px;
          margin-bottom: 10px;
        }
        .w60 {
          width: 40%;
          height: 12px;
        }
        .w40 {
          width: 55%;
        }
        .w100 {
          width: 100%;
        }
        .h28 {
          height: 26px;
        }
        .h8 {
          height: 8px;
        }
        @keyframes shimmer {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }
      `}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// View 2 — Prediction History
// ----------------------------------------------------------------------------

function HistoryView({ onError }: { onError: (msg: string) => void }) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(
    async (isRefresh = false) => {
      isRefresh ? setRefreshing(true) : setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/v1/predictions?limit=10`, {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const data = await res.json();
        setItems(Array.isArray(data.items) ? data.items : []);
      } catch (err) {
        onError(
          err instanceof Error
            ? `Could not load history: ${err.message}`
            : "Could not load prediction history."
        );
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [onError]
  );

  useEffect(() => {
    load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section aria-label="Prediction history">
      <div className="head">
        <div>
          <h2>Prediction History</h2>
          <p className="hint">Most recent 10 inference runs</p>
        </div>
        <button className="refresh" onClick={() => load(true)} disabled={refreshing}>
          <RefreshCw size={15} strokeWidth={2} className={refreshing ? "spin" : ""} />
          Refresh
        </button>
      </div>

      <div className="card table-card">
        {loading ? (
          <HistorySkeleton />
        ) : items.length === 0 ? (
          <div className="empty">
            <Inbox size={26} strokeWidth={1.5} />
            <p>No predictions yet</p>
            <span>Run a classification to see it appear here.</span>
          </div>
        ) : (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Image</th>
                  <th>Predicted Class</th>
                  <th>Confidence</th>
                  <th>Latency</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <tr key={it.id}>
                    <td data-label="ID" className="mono">
                      {it.id}
                    </td>
                    <td data-label="Image">{it.image_name}</td>
                    <td data-label="Predicted Class">
                      <span
                        className="chip"
                        style={{ borderColor: accentForClass(it.predicted_class) }}
                      >
                        {it.predicted_class}
                      </span>
                    </td>
                    <td data-label="Confidence" className="mono">
                      {pct(it.confidence)}
                    </td>
                    <td data-label="Latency" className="mono">
                      {Math.round(it.inference_ms)} ms
                    </td>
                    <td data-label="Timestamp" className="mono muted">
                      {formatTimestamp(it.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <style jsx>{`
        .head {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          margin-bottom: 16px;
          gap: 12px;
        }
        h2 {
          font-family: var(--font-display);
          font-weight: 500;
          font-size: 20px;
        }
        .hint {
          font-size: 12.5px;
          color: var(--slate);
          margin-top: 2px;
        }
        .refresh {
          display: flex;
          align-items: center;
          gap: 6px;
          background: var(--charcoal);
          color: #fff;
          border: none;
          padding: 9px 16px;
          border-radius: 100px;
          font-size: 13px;
          font-weight: 500;
          flex-shrink: 0;
          box-shadow: 0 4px 12px rgba(32, 31, 26, 0.18);
          transition: background 0.25s ease, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
        }
        .refresh:hover:not(:disabled) {
          background: var(--slate-dark);
          transform: translateY(-1px);
          box-shadow: 0 8px 18px rgba(32, 31, 26, 0.24);
        }
        .refresh:active:not(:disabled) {
          transform: translateY(0) scale(0.97);
        }
        .refresh:disabled {
          opacity: 0.7;
        }
        .spin {
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
        .card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 16px;
          overflow: hidden;
        }
        .table {
          width: 100%;
          border-collapse: collapse;
        }
        thead th {
          text-align: left;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--slate);
          font-family: var(--font-mono);
          padding: 14px 20px;
          border-bottom: 1px solid var(--line);
        }
        tbody td {
          padding: 14px 20px;
          font-size: 13.5px;
          border-bottom: 1px solid var(--line);
          color: var(--charcoal);
        }
        tbody tr:last-child td {
          border-bottom: none;
        }
        tbody tr:hover {
          background: rgba(28, 33, 40, 0.025);
        }
        .mono {
          font-family: var(--font-mono);
        }
        .muted {
          color: var(--slate);
        }
        .chip {
          display: inline-block;
          font-size: 12px;
          font-weight: 600;
          padding: 3px 10px;
          border-radius: 100px;
          border: 1.5px solid;
        }
        .empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          gap: 6px;
          color: var(--slate);
          padding: 60px 20px;
        }
        .empty p {
          font-size: 14px;
          font-weight: 600;
          color: var(--charcoal);
        }
        .empty span {
          font-size: 12.5px;
        }
        @media (max-width: 720px) {
          thead {
            display: none;
          }
          .table,
          tbody,
          tr,
          td {
            display: block;
            width: 100%;
          }
          tbody tr {
            padding: 14px 18px;
          }
          tbody td {
            border: none;
            padding: 4px 0;
            display: flex;
            justify-content: space-between;
            gap: 12px;
          }
          tbody td::before {
            content: attr(data-label);
            font-family: var(--font-mono);
            font-size: 10.5px;
            text-transform: uppercase;
            color: var(--slate);
            letter-spacing: 0.05em;
          }
        }
      `}</style>
    </section>
  );
}

function HistorySkeleton() {
  return (
    <div className="rows">
      {[0, 1, 2, 3, 4].map((i) => (
        <div className="row" key={i}>
          <div className="s w10" />
          <div className="s w20" />
          <div className="s w20" />
          <div className="s w12" />
          <div className="s w12" />
          <div className="s w16" />
        </div>
      ))}
      <style jsx>{`
        .rows {
          padding: 8px 20px;
        }
        .row {
          display: flex;
          gap: 20px;
          padding: 13px 0;
          border-bottom: 1px solid var(--line);
        }
        .row:last-child {
          border-bottom: none;
        }
        .s {
          height: 12px;
          border-radius: 6px;
          background: linear-gradient(90deg, #eceff0 0%, #f6f7f7 50%, #eceff0 100%);
          background-size: 200% 100%;
          animation: shimmer 1.4s ease-in-out infinite;
        }
        .w10 {
          width: 6%;
        }
        .w12 {
          width: 10%;
        }
        .w16 {
          width: 14%;
        }
        .w20 {
          width: 20%;
        }
        @keyframes shimmer {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }
      `}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// View 3 — Operational Stats
// ----------------------------------------------------------------------------

function StatsView({ onError }: { onError: (msg: string) => void }) {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" });
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const data: StatsData = await res.json();
        if (!cancelled) setStats(data);
      } catch (err) {
        if (!cancelled) {
          onError(
            err instanceof Error
              ? `Could not load stats: ${err.message}`
              : "Could not load operational stats."
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [onError]);

  const distEntries = stats
    ? Object.entries(stats.class_distribution).sort((a, b) => b[1] - a[1])
    : [];
  const distTotal = distEntries.reduce((sum, [, c]) => sum + c, 0);

  return (
    <section aria-label="Operational analytics">
      <h2>Operational Stats</h2>
      <p className="hint">Live aggregate metrics from the inference agent</p>

      {loading ? (
        <StatsSkeleton />
      ) : !stats ? (
        <div className="card empty">
          <Inbox size={26} strokeWidth={1.5} />
          <p>No stats available yet</p>
        </div>
      ) : (
        <>
          <div className="metrics">
            <div className="metric-card">
              <span className="m-label">Total Predictions</span>
              <span className="m-value">{stats.total_predictions.toLocaleString()}</span>
            </div>
            <div className="metric-card">
              <span className="m-label">Average Confidence</span>
              <span className="m-value">{pct(stats.average_confidence)}</span>
            </div>
          </div>

          <div className="card dist-card">
            <h3>Class Distribution</h3>
            {distEntries.length === 0 ? (
              <p className="dist-empty">No class distribution data yet.</p>
            ) : (
              distEntries.map(([name, count]) => {
                const share = distTotal > 0 ? count / distTotal : 0;
                return (
                  <div className="dist-row" key={name}>
                    <div className="dist-top">
                      <span className="dist-name">{name}</span>
                      <span className="dist-count">
                        {count} · {pct(share)}
                      </span>
                    </div>
                    <div className="dist-track">
                      <div
                        className="dist-fill"
                        style={{ width: pct(share), background: accentForClass(name) }}
                      />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </>
      )}

      <style jsx>{`
        h2 {
          font-family: var(--font-display);
          font-weight: 500;
          font-size: 20px;
        }
        .hint {
          font-size: 12.5px;
          color: var(--slate);
          margin-top: 2px;
          margin-bottom: 20px;
        }
        .metrics {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
          margin-bottom: 20px;
        }
        @media (max-width: 560px) {
          .metrics {
            grid-template-columns: 1fr;
          }
        }
        .metric-card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 22px 24px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .m-label {
          font-family: var(--font-mono);
          font-size: 11.5px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--slate);
        }
        .m-value {
          font-family: var(--font-display);
          font-size: 34px;
          font-weight: 600;
          color: var(--charcoal);
        }
        .card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 24px;
        }
        h3 {
          font-family: var(--font-display);
          font-weight: 500;
          font-size: 16px;
          margin-bottom: 16px;
        }
        .dist-row {
          margin-bottom: 16px;
        }
        .dist-row:last-child {
          margin-bottom: 0;
        }
        .dist-top {
          display: flex;
          justify-content: space-between;
          margin-bottom: 6px;
          font-size: 13px;
        }
        .dist-name {
          font-weight: 600;
          color: var(--charcoal);
        }
        .dist-count {
          font-family: var(--font-mono);
          font-size: 12px;
          color: var(--slate);
        }
        .dist-track {
          height: 8px;
          border-radius: 100px;
          background: rgba(28, 33, 40, 0.06);
          overflow: hidden;
        }
        .dist-fill {
          height: 100%;
          border-radius: 100px;
          transition: width 0.9s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .dist-empty {
          font-size: 13px;
          color: var(--slate);
        }
        .empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          color: var(--slate);
          padding: 50px 20px;
          text-align: center;
        }
      `}</style>
    </section>
  );
}

function StatsSkeleton() {
  return (
    <div>
      <div className="metrics">
        <div className="s tall" />
        <div className="s tall" />
      </div>
      <div className="s block" />
      <style jsx>{`
        .metrics {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
          margin-bottom: 20px;
        }
        @media (max-width: 560px) {
          .metrics {
            grid-template-columns: 1fr;
          }
        }
        .s {
          background: linear-gradient(90deg, #eceff0 0%, #f6f7f7 50%, #eceff0 100%);
          background-size: 200% 100%;
          animation: shimmer 1.4s ease-in-out infinite;
          border-radius: 16px;
        }
        .tall {
          height: 96px;
        }
        .block {
          height: 220px;
        }
        @keyframes shimmer {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }
      `}</style>
    </div>
  );
}

// ----------------------------------------------------------------------------
// View 4 — Chat with Open WebUI
// ----------------------------------------------------------------------------

function ChatView() {
  return (
    <section aria-label="Chat with Open WebUI">
      <div className="chat-container">
        <div className="chat-placeholder">
          <MessageCircle size={48} strokeWidth={1} />
          <h3>Open WebUI Chat</h3>
          <p>Open WebUI is initializing. Click the button below to access the chat interface.</p>
          <button
            className="open-button"
            onClick={() => window.open("http://localhost:8080", "_blank")}
          >
            Open Chat in New Window
          </button>
          <p className="hint">Chat will open at http://localhost:8080</p>
        </div>
      </div>

      <style jsx>{`
        section {
          height: 100%;
          display: flex;
          flex-direction: column;
        }
        .chat-container {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: var(--card);
          padding: 40px;
        }
        .chat-placeholder {
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
          color: var(--slate);
        }
        .chat-placeholder svg {
          color: var(--accent);
          opacity: 0.7;
        }
        .chat-placeholder h3 {
          font-family: var(--font-display);
          font-weight: 500;
          font-size: 20px;
          color: var(--charcoal);
          margin: 0;
        }
        .chat-placeholder p {
          margin: 0;
          font-size: 14px;
          color: var(--slate);
          max-width: 320px;
        }
        .open-button {
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 10px;
          padding: 12px 24px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: background 0.3s ease, transform 0.2s ease;
          margin-top: 8px;
        }
        .open-button:hover {
          background: var(--cyanotype-deep);
          transform: translateY(-2px);
        }
        .open-button:active {
          transform: translateY(0);
        }
        .hint {
          font-size: 12px !important;
          color: var(--mist) !important;
          margin-top: 8px;
        }
      `}</style>
    </section>
  );
}
