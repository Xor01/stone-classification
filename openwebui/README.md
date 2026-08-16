# Open WebUI Integration Guide

This guide explains how to connect **Open WebUI** to the production Computer Vision backend and agent tools.

## 1. Why `Failed to resolve 'backend'` Occurred

If Open WebUI is running inside a Docker container while your FastAPI backend is running locally on your host machine (e.g. via `uvicorn`), the container cannot resolve `backend` or `localhost`. 

- **If Open WebUI is in Docker and backend is on host**: Use `http://host.docker.internal:8000`
- **If both run locally without Docker**: Use `http://localhost:8000`
- **If running full stack via Docker Compose**: Use `http://backend:8000`

---

## 2. Updated Tool Code with `Valves` & Auto-Fallback

The updated [`openwebui_tools.py`](./openwebui_tools.py) includes Open WebUI `Valves` (which lets you configure the URL directly in Open WebUI's UI settings) and **automatic fallback probing** across `http://host.docker.internal:8000`, `http://localhost:8000`, and `http://backend:8000`.

### Copy & Paste into Open WebUI:
1. In Open WebUI, navigate to **Workspace** > **Tools** > Select your tool (or click **+ Create Tool**).
2. Replace the tool code with the code from [`openwebui_tools.py`](./openwebui_tools.py).
3. Click **Save**.
4. In the tool settings gear icon (Valves), verify the `BACKEND_URL` (default is `http://host.docker.internal:8000`).

---

## 3. Test Prompts in Open WebUI

Ensure the tool is toggled ON for your model/chat, then ask:
* *"Classify this image: https://example.com/painting.jpg"* or *"Classify the image at /path/to/image.jpg"*
* *"Which computer vision model is currently deployed?"*
* *"Show me the latest 3 predictions."*
* *"What are the prediction statistics?"*

