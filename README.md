# Geological Rock Classification & Multimodal AI Agent Platform

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(ConvNeXt--Tiny)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15%2B%20(React%2019)-000000.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Observability%20%26%20Tracing-FF6B6B.svg)](https://langfuse.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Full%20Stack-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Package Manager - uv](https://img.shields.io/badge/uv-Fast%20Python%20Packaging-2B5797.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, full-stack **Geological Rock Classification & Multimodal AI System**. The platform integrates a high-performance **ConvNeXt-Tiny** deep learning model, an asynchronous **FastAPI + PostgreSQL** backend, an intelligent **LangChain** tool-calling agent with **Langfuse** observability, a modern **Next.js** web interface with real-time analytics and voice interaction (Whisper STT / OpenAI TTS), and containerized **Open WebUI + Pipelines** integration.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Target Lithologies & Dataset](#-target-lithologies--dataset)
- [Model Engineering & Computer Vision Core](#-model-engineering--computer-vision-core)
- [API Reference & Service Endpoints](#-api-reference--service-endpoints)
- [Intelligent Agent & Observability (Langfuse)](#-intelligent-agent--observability-langfuse)
- [Frontend Web Studio & Multimodal Voice](#-frontend-web-studio--multimodal-voice)
- [Open WebUI & Pipelines Integration](#-open-webui--pipelines-integration)
- [Getting Started & Local Development](#-getting-started--local-development)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Running with Docker Compose (Recommended)](#running-with-docker-compose-recommended)
  - [Running Services Locally](#running-services-locally)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🏛️ System Architecture

The platform is designed around a modular microservice architecture combining offline model training, resilient REST services, persistent relational storage, conversational agent intelligence, and interactive client applications:

```mermaid
flowchart TD
    subgraph Data_Model_Pipeline ["🔬 Computer Vision & Training Subsystem"]
        A[Roboflow Dataset<br/>4,212 Rock Samples] --> B[Data Preprocessing & Augmentation<br/>transforms.py / dataset.py]
        B --> C[ConvNeXt-Tiny Transfer Learning<br/>AdamW / CrossEntropyLoss]
        C --> D[(Exported Artifacts<br/>models/model.pt & labels.json)]
    end

    subgraph Backend_Core ["⚡ FastAPI & PostgreSQL Backend"]
        D --> E[Inference Service<br/>Singleton Model Loader]
        F[(PostgreSQL 16<br/>predictions table)] <--> G[Prediction Service & SQLModel ORM]
        E --> G
        G --> H[FastAPI REST API Routes<br/>/predict, /history, /stats, /model, /health]
        I[Voice Service<br/>Whisper STT / OpenAI TTS] --> H
        J[Attachment Sandbox<br/>Secure Image Handler] --> H
    end

    subgraph Agentic_Layer ["🤖 Agentic AI & Observability"]
        K[LangChain Agent Engine<br/>create_agent + gpt-4o-mini] <--> L[Agent Tools<br/>classify_image, stats, history, model_info]
        L <--> H
        K --> M[Langfuse Tracing<br/>Cloud Traces, Latency & Token Metrics]
        H -.->|Direct Prediction Spans| M
    end

    subgraph Frontend_Clients ["🌐 User Interface & Client Apps"]
        N[Next.js 15 Web Dashboard<br/>Classification, Analytics, Voice Chat] <--> H
        N <--> K
        O[Open WebUI Assistant<br/>Custom OpenWebUI Tools] <--> P[Pipelines Filter<br/>Langfuse v3 Pipeline]
        P <--> H
        P --> M
    end
```

---

## ✨ Key Features

- **🚀 State-of-the-Art Deep Learning (ConvNeXt-Tiny)**:
  - ImageNet-pretrained backbone with fine-tuned classification top (`Dropout(0.3) + Linear(in_features, 9)`).
  - Robust preprocessing with ImageNet normalization and dynamic augmentation (rotations, flips, color jitter).
  - High-confidence multi-class rock identification across 9 canonical lithological categories.

- **⚡ Production-Ready FastAPI Backend**:
  - Asynchronous endpoints for single and batch predictions, history exploration, and aggregate telemetry.
  - Persistent storage using **PostgreSQL 16** and **SQLModel** ORM.
  - Full input validation, image type sanitization (`JPEG`, `PNG`, `WEBP`), size enforcement (up to 15MB), and fail-fast health probes.

- **🤖 Tool-Augmented Conversational AI Agent**:
  - Powered by **LangChain** and modern LLMs (`gpt-4o-mini`, Gemini, or Claude).
  - Equips the LLM with backend-integrated tools: `classify_image`, `get_model_info`, `get_prediction_history`, `get_prediction_by_id`, and `get_prediction_statistics`.
  - System prompt enforcing strict anti-hallucination policies and ground-truth reporting.

- **🎙️ Multimodal Voice & Audio Support**:
  - **Speech-to-Text (STT)**: Direct audio transcription powered by OpenAI Whisper (`whisper-1`).
  - **Text-to-Speech (TTS)**: High-quality natural voice synthesis (`tts-1` with configurable voices like `alloy`).
  - Seamless hands-free voice conversations in the web UI.

- **📊 Comprehensive Observability (Langfuse)**:
  - Automatic end-to-end tracing of all agent conversations, tool invocations, and direct `/predict` API spans.
  - Environment tagging (`development` vs `production`) for metric isolation, user/session attribution, and latency tracking.

- **💻 Modern Next.js Web Studio**:
  - **Live Classification Studio**: Instant drag-and-drop image uploads, confidence progress gauges, top-5 prediction breakdowns, and millisecond latency timers.
  - **Historical Explorer**: Filterable, paginated audit log of all historical classifications with thumbnail previews.
  - **Analytics Dashboard**: Dynamic metrics showing total volume, average model confidence, and class distribution.
  - **Interactive AI Assistant Panel**: Chat interface with attachment uploads, voice recording, and audio playback.

- **🐳 Docker Compose Deployment**:
  - Single-command orchestration for the entire stack: PostgreSQL database, FastAPI backend, Next.js frontend, Open WebUI, and Pipelines.

---

## 📁 Repository Structure

```
stone-classification/
├── agent/                          # LangChain Agent Subsystem
│   ├── agent.py                    # LLM instantiation, Langfuse handler & run_agent runner
│   ├── prompts.py                  # System prompt and domain rules
│   └── tools.py                    # LangChain tool bindings (classify, history, stats, model info)
├── backend/                        # FastAPI Backend Service
│   ├── app/
│   │   ├── api/                    # REST API Endpoints
│   │   │   ├── agent_api.py        # POST /api/v1/agent/chat
│   │   │   ├── attachments.py      # POST /api/v1/attachments (sandboxed uploads)
│   │   │   ├── health.py           # GET /health
│   │   │   ├── history.py          # GET /api/v1/predictions, GET /api/v1/predictions/{id}
│   │   │   ├── model_info.py       # GET /api/v1/model
│   │   │   ├── predictions.py      # POST /api/v1/predict
│   │   │   ├── stats.py            # GET /api/v1/stats
│   │   │   └── voice.py            # POST /api/v1/voice/transcribe, POST /api/v1/voice/speak
│   │   ├── models/                 # Deployed Model Artifacts
│   │   │   ├── labels.json         # Index-to-class mapping
│   │   │   └── model.pt            # PyTorch state_dict (~106MB)
│   │   ├── services/               # Core Business Logic
│   │   │   ├── attachments.py      # Attachment directory validation
│   │   │   ├── inference.py        # ConvNeXt-Tiny PyTorch inference singleton
│   │   │   ├── prediction_service.py # DB read/write operations
│   │   │   └── speech.py           # Whisper STT & OpenAI TTS service
│   │   ├── config.py               # Pydantic Settings & environment variables
│   │   ├── database.py             # PostgreSQL engine & session setup
│   │   ├── main.py                 # FastAPI application factory & error middleware
│   │   ├── models.py               # SQLModel table definition (predictions)
│   │   ├── observability.py        # Langfuse tracing helpers & context managers
│   │   └── schemas.py              # Pydantic request/response validation schemas
│   ├── tests/                      # Automated Pytest Suite (health, predict, agent, voice, etc.)
│   ├── Dockerfile                  # Production container definition for backend
│   └── pyproject.toml              # Backend dependencies & configuration
├── data/                           # Dataset Metadata & Raw Data
│   ├── README.dataset.txt          # Roboflow dataset citation
│   └── README.roboflow.txt         # Dataset version and preprocessing metadata
├── frontend/                       # Web Client Application
│   └── stone-classification-openui/ # Next.js 15 App Router Frontend
│       ├── src/
│       │   ├── app/                # Next.js Pages, Layouts & Route Handlers
│       │   ├── components/         # React Components (AgentChat, CloudChat, UI elements)
│       │   ├── hooks/              # Custom Hooks (useVoice, usePersistedModel)
│       │   └── lib/                # Client utilities & model switcher definitions
│       ├── Dockerfile              # Production container definition for frontend
│       └── package.json            # Node.js dependencies & scripts
├── openwebui/                      # Open WebUI Integration
│   ├── openwebui_tools.py          # Custom Open WebUI Tool definitions with auto-probing Valves
│   └── README.md                   # Open WebUI connection & setup guide
├── scripts/                        # Automation & Diagnostic Scripts
│   └── test_langfuse.py            # Diagnostic script to verify Langfuse keys and tracing
├── training/                       # Computer Vision Model Training Pipeline
│   ├── dataset.py                  # PyTorch Dataset and DataLoader loaders
│   ├── evaluate.py                 # Test set evaluation & classification report generator
│   ├── train.py                    # PyTorch ConvNeXt-Tiny training & checkpointing loop
│   ├── transforms.py               # Training and validation image transformation pipelines
│   └── pyproject.toml              # Training dependencies
├── compose.yaml                    # Local multi-service Docker Compose configuration
├── docker-compose.yml              # Production/Deployment Docker Compose configuration
├── pyproject.toml                  # Root project workspace configuration (uv)
└── README.md                       # Main Project Documentation
```

---

## 📊 Target Lithologies & Dataset

The geological dataset is curated from [Roboflow Universe: Rock Classification](https://universe.roboflow.com/william-cwsr-8jizi/rock-clasfication), containing **4,212 high-resolution geological specimens** divided across 3 canonical splits:

| Split | Image Count | Percentage | Pipeline Function |
| :--- | :---: | :---: | :--- |
| **Train** | **3,687** | 87.5% | Feature extraction and gradient updates |
| **Validation** | **351** | 8.3% | Checkpoint evaluation and hyperparameter tuning |
| **Test** | **174** | 4.1% | Out-of-sample generalization benchmark |
| **Total** | **4,212** | 100.0% | Complete rock dataset |

### Target Classes (9 Lithologies)

```json
{
  "0": "Basalt",
  "1": "Clay",
  "2": "Conglomerate",
  "3": "Diatomite",
  "4": "Shale-(Mudstone)",
  "5": "Siliceous-sinter",
  "6": "chert",
  "7": "gypsum",
  "8": "olivine-basalt"
}
```

- **`Basalt`** *(Igneous - Extrusive)*: Dark-colored, fine-grained mafic volcanic rock composed primarily of plagioclase and pyroxene.
- **`Clay`** *(Sedimentary - Argillaceous)*: Fine-grained natural rock or soil material combining one or more clay minerals.
- **`Conglomerate`** *(Sedimentary - Clastic)*: Coarse-grained sedimentary rock composed of rounded gravel and pebble clasts.
- **`Diatomite`** *(Sedimentary - Biogenic)*: Soft, crumbly, siliceous sedimentary rock derived from fossilized diatom remains.
- **`Shale-(Mudstone)`** *(Sedimentary - Clastic)*: Fine-grained, laminated rock formed by the consolidation of clay, silt, or mud.
- **`Siliceous-sinter`** *(Sedimentary - Chemical)*: Porous, low-density opaline silica deposit formed around hot springs and geysers.
- **`chert`** *(Sedimentary - Chemical/Biogenic)*: Hard, microcrystalline quartz rock exhibiting sharp conchoidal fracturing.
- **`gypsum`** *(Sedimentary - Evaporite)*: Soft sulfate mineral ($CaSO_4 \cdot 2H_2O$) with high albedo and crystalline/fibrous structures.
- **`olivine-basalt`** *(Igneous - Mafic/Ultramafic)*: Basaltic groundmass enriched with distinctive vitreous olive-green olivine phenocrysts.

---

## 🔬 Model Engineering & Computer Vision Core

### Architecture & Preprocessing
The vision subsystem utilizes **ConvNeXt-Tiny** with ImageNet-1k pre-trained weights. The classifier top is adapted for the 9 target classes:

$$\text{Input } (224 \times 224 \times 3) \longrightarrow \text{ConvNeXt-Tiny Backbone} \longrightarrow \text{Dropout}(p=0.3) \longrightarrow \text{Linear}(768 \to 9)$$

- **Training Augmentation** (`training/transforms.py`):
  - Random Horizontal & Vertical Flips ($p=0.5$)
  - Random Affine Rotation ($\pm 45^\circ$)
  - Color Jitter (Brightness $\pm 10\%$, Contrast $\pm 10\%$)
  - Standard ImageNet Normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$)
- **Validation / Inference Transform**:
  - Deterministic Bicubic Resize to $224 \times 224$ pixels followed by ImageNet Normalization.
- **Optimization Strategy**:
  - Optimizer: **AdamW** ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-2}$)
  - Loss Function: **Cross-Entropy Loss** with Softmax probability outputs

### Reproduction Commands
```bash
# Train the ConvNeXt-Tiny model
uv run python training/train.py

# Evaluate test set accuracy and generate reports/model_metrics.json
uv run python training/evaluate.py
```

---

## 📡 API Reference & Service Endpoints

The FastAPI server provides interactive OpenAPI documentation at `/docs` (Swagger UI) and `/redoc`.

| Method | Endpoint | Description | Auth / Input |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness & readiness probe verifying DB and model status | None |
| `POST` | `/api/v1/predict` | Classifies an uploaded image and saves prediction to DB | Multipart `image` file (`JPEG/PNG/WEBP`) |
| `GET` | `/api/v1/predictions` | Retrieves paginated historical predictions | Query params: `limit` (default: 20), `offset` (default: 0) |
| `GET` | `/api/v1/predictions/{id}`| Retrieves details of a specific prediction by ID | Path param: `id` (integer) |
| `GET` | `/api/v1/stats` | Aggregates summary statistics (total, avg confidence, distribution) | None |
| `GET` | `/api/v1/model` | Returns deployed model metadata, labels, and version | None |
| `POST` | `/api/v1/agent/chat` | Natural language chat endpoint for the AI Agent | JSON: `{ "message": str, "session_id"?: str, "attachment_path"?: str }` |
| `POST` | `/api/v1/attachments` | Uploads an image attachment to the secure sandbox | Multipart `file` |
| `POST` | `/api/v1/voice/transcribe` | Transcribes spoken audio to text via Whisper | Multipart audio file (`webm/wav/mp3`) |
| `POST` | `/api/v1/voice/speak` | Converts text to synthesized speech via OpenAI TTS | JSON: `{ "text": str }` -> Returns `audio/mpeg` |

### Sample Prediction Response (`POST /api/v1/predict`)
```json
{
  "id": 42,
  "predicted_class": "olivine-basalt",
  "confidence": 0.9412,
  "top_predictions": [
    { "class_name": "olivine-basalt", "probability": 0.9412 },
    { "class_name": "Basalt", "probability": 0.0418 },
    { "class_name": "chert", "probability": 0.0095 },
    { "class_name": "Siliceous-sinter", "probability": 0.0041 },
    { "class_name": "Conglomerate", "probability": 0.0034 }
  ],
  "inference_ms": 32.45,
  "model_version": "1.0.0",
  "request_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "created_at": "2026-08-18T11:20:00Z"
}
```

---

## 🤖 Intelligent Agent & Observability (Langfuse)

The agent layer in `agent/` uses LangChain tools to inspect the real system state:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          LANGCHAIN AGENT TOOLS                         │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Tool Name                │ Functionality                               │
├──────────────────────────┼─────────────────────────────────────────────┤
│ `classify_image`         │ Invokes ConvNeXt-Tiny inference on image    │
│ `get_model_info`         │ Fetches active model version and classes    │
│ `get_prediction_history` │ Queries database for recent predictions     │
│ `get_prediction_by_id`   │ Looks up a single record by primary key     │
│ `get_prediction_statistics`│ Retrieves total counts & class distribution│
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Full-Stack Tracing with Langfuse
1. **Agent Chat Tracing**: Automatically records user prompts, LLM thought chains, tool calls, token usage, and latency.
2. **Direct API Classification Spans**: `POST /api/v1/predict` records inference time, image metadata, and predicted classes into Langfuse without exposing customer image bytes.
3. **Environment Segregation**: Traces carry the `ENV` tag (`development`, `staging`, `production`) to maintain clean metric separation.

```bash
# Verify Langfuse authentication and agent tracing locally:
uv run python scripts/test_langfuse.py
```

---

## 💻 Frontend Web Studio & Multimodal Voice

The frontend located in `frontend/stone-classification-openui/` offers a rich user experience built with **Next.js 15** and **React 19**:

- **Interactive Classification Hub**: Instant preview on drag-and-drop, dynamic probability distribution bars with color-coded confidence indicators, and sub-50ms latency feedback.
- **Auditing & History Table**: Live chronological feed of past classifications with modal zoom and class filtering.
- **Analytics Visualizer**: Interactive statistics showing class distribution percentages and dataset volume.
- **Multimodal AI Companion**:
  - Voice Recording with live audio wave visualization and direct transcription via Whisper.
  - Text-to-Speech audio streaming of the agent's geological explanations.
  - Image attachment upload button allowing users to ask: *"What kind of rock is this and what minerals formed it?"*

---

## 🔌 Open WebUI & Pipelines Integration

The repository includes a ready-to-use toolset for **Open WebUI** located in `openwebui/`:

1. **Tool Definition (`openwebui/openwebui_tools.py`)**:
   - Supports Open WebUI **Valves** for dynamic endpoint configuration.
   - Built-in multi-host fallback probing (`http://host.docker.internal:8000`, `http://localhost:8000`, `http://backend:8000`).
   - Integrated Langfuse observability directly from Open WebUI tool executions.
2. **Pipelines Filter**:
   - Docker Compose connects Open WebUI to `ghcr.io/open-webui/pipelines` loaded with the Langfuse v3 filter pipeline.

---

## 🚀 Getting Started & Local Development

### Prerequisites
- **Python**: `3.11+` (Python 3.13 recommended)
- **Node.js**: `20.x+` and `pnpm` or `npm`
- **Docker & Docker Compose**: For containerized deployment
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (ultra-fast Python package manager)

### Environment Configuration

Create your local `.env` configuration file in the project root:

```bash
cp backend/.env.example .env
```

Ensure the following environment variables are configured:

```env
# Application Settings
ENV=development
DEBUG=true
BACKEND_PORT=8000
FRONTEND_PORT=3000
OPENWEBUI_PORT=8080
PIPELINES_PORT=9099

# Database Configuration
POSTGRES_USER=cvuser
POSTGRES_PASSWORD=cvpassword
POSTGRES_DB=cvapp
POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://cvuser:cvpassword@localhost:5432/cvapp

# Model Configuration
MODEL_PATH=app/models/model.pt
LABELS_PATH=app/models/labels.json
MODEL_VERSION=1.0.0
MAX_UPLOAD_MB=8
ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/webp

# LLM & OpenAI Configuration
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

### Running with Docker Compose (Recommended)

To build and spin up the complete multi-service stack with a single command:

```bash
# Build and start all services (PostgreSQL, Backend, Frontend, Open WebUI, Pipelines)
docker compose -f compose.yaml up --build -d

# View real-time logs
docker compose -f compose.yaml logs -f
```

Once running, access the services:
- 🌐 **Web Studio UI**: [http://localhost:3000](http://localhost:3000)
- ⚡ **FastAPI Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 💬 **Open WebUI Assistant**: [http://localhost:8080](http://localhost:8080)
- 📊 **Langfuse Observability**: [https://cloud.langfuse.com](https://cloud.langfuse.com)

To stop the containers:
```bash
docker compose -f compose.yaml down
```

---

### Running Services Locally

#### 1. Start PostgreSQL
```bash
docker run --name stone-db -e POSTGRES_USER=cvuser -e POSTGRES_PASSWORD=cvpassword -e POSTGRES_DB=cvapp -p 5432:5432 -d postgres:16-alpine
```

#### 2. Start the FastAPI Backend
```bash
# Sync dependencies using uv
uv sync

# Run backend server
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Start the Next.js Frontend
```bash
cd frontend/stone-classification-openui

# Install Node dependencies
npm install

# Start Next.js development server
npm run dev
```

---

## 🧪 Testing & Quality Assurance

The backend includes a comprehensive **pytest** test suite covering API contracts, database persistence, image validation, agent tools, voice handling, and attachment sandboxing:

```bash
# Run the test suite from the backend directory
cd backend
uv run pytest -v
```

### Test Coverage Highlights:
- `test_health.py`: Verifies `/health` endpoint status code and JSON schema.
- `test_predictions.py`: Validates image classification, Top-K probability ranking, size limits, and invalid MIME type rejection.
- `test_database_insert.py`: Confirms SQLModel prediction persistence and timestamp integrity.
- `test_agent.py`: Mocks and validates LangChain agent tools, execution flows, and failure handling.
- `test_attachments.py` & `test_attachments_api.py`: Validates sandboxed directory traversal protections.
- `test_speech.py` & `test_voice_api.py`: Verifies STT transcription and TTS audio synthesis endpoints.

---

## 📜 License & Acknowledgments

- **Dataset**: Provided by [Roboflow Universe user `william-cwsr-8jizi`](https://universe.roboflow.com/william-cwsr-8jizi/rock-clasfication) under the **MIT License**.
- **Model Architecture**: [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) by Meta AI Research.
- **License**: This project is open-source and released under the **MIT License**.
