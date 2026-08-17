"""
title: CV System Agent Tools
author: AI Agentic Engineer
version: 1.5.0
description: Tools for Open WebUI to communicate with the deployed Computer Vision FastAPI backend with built-in Langfuse tracing.
requirements: langfuse>=4.14.4
"""

import base64
import io
import os
import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        BACKEND_URL: str = Field(
            default=os.getenv("BACKEND_URL", "http://host.docker.internal:8000"),
            description="Base URL of FastAPI backend. Use 'http://host.docker.internal:8000' if Open WebUI is in Docker and backend is on host, 'http://localhost:8000' for local dev, or 'http://backend:8000' in Docker Compose.",
        )
        LANGFUSE_PUBLIC_KEY: str = Field(
            default=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            description="Langfuse Public Key for observability tracing.",
        )
        LANGFUSE_SECRET_KEY: str = Field(
            default=os.getenv("LANGFUSE_SECRET_KEY", ""),
            description="Langfuse Secret Key for observability tracing.",
        )
        LANGFUSE_HOST: str = Field(
            default=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            description="Langfuse Host URL.",
        )


    def __init__(self):
        self.valves = self.Valves()
        self._langfuse = None

    def _get_langfuse(self):
        """Build (once) the Langfuse client, or return None if unconfigured."""
        if self._langfuse is not None:
            return self._langfuse

        pub_key = (self.valves.LANGFUSE_PUBLIC_KEY or "").strip("\"' ")
        sec_key = (self.valves.LANGFUSE_SECRET_KEY or "").strip("\"' ")
        host = (self.valves.LANGFUSE_HOST or "https://cloud.langfuse.com").strip("\"' ")

        if not pub_key or not sec_key:
            print("[cv-agent-tools] Langfuse keys not set; tracing disabled.")
            return None

        from langfuse import Langfuse

        self._langfuse = Langfuse(
            public_key=pub_key, secret_key=sec_key, host=host
        )
        return self._langfuse

    def _log_langfuse(self, tool_name: str, input_data: any, output_data: any):
        try:
            lf = self._get_langfuse()
            if lf is None:
                return

            from langfuse import propagate_attributes

            # propagate_attributes sets the trace-level name/tags; the tool
            # observation itself carries the input/output payload.
            with propagate_attributes(
                trace_name=f"tool:{tool_name}",
                tags=["openwebui", "cv-agent-tool"],
                environment=os.getenv("ENV", "development"),
            ):
                with lf.start_as_current_observation(
                    name=f"tool:{tool_name}",
                    as_type="tool",
                    input=input_data,
                    output=output_data,
                ):
                    pass
            lf.flush()
        except Exception as e:
            # Never fail a tool call because tracing broke, but do not hide it
            # either - a silent `pass` here masked a total tracing outage.
            print(f"[cv-agent-tools] Langfuse logging failed: {type(e).__name__}: {e}")

    def _get_candidate_urls(self):
        urls = [self.valves.BACKEND_URL]
        for fallback in [
            "http://host.docker.internal:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://backend:8000",
        ]:
            if fallback not in urls:
                urls.append(fallback)
        return urls

    def _request(self, method: str, endpoint: str, **kwargs):
        candidate_urls = self._get_candidate_urls()
        last_error = None
        for base in candidate_urls:
            url = f"{base.rstrip('/')}{endpoint}"
            try:
                r = requests.request(method, url, timeout=30, **kwargs)
                r.raise_for_status()
                return r.text
            except Exception as e:
                last_error = e
                continue
        raise last_error if last_error else Exception("Failed to connect to backend")

    def _find_image_bytes(self, image_source: str) -> tuple[str, bytes]:
        """Resolve an image source (URL, Open WebUI file ID, file path, base64) to (filename, bytes)."""
        image_source = str(image_source).strip().strip("'\"")

        # 1. Direct Web URL
        if image_source.startswith(("http://", "https://")):
            r = requests.get(image_source, timeout=15)
            r.raise_for_status()
            fname = os.path.basename(image_source.split("?")[0]) or "downloaded.jpg"
            return fname, r.content

        # 2. Base64 Data URI
        if image_source.startswith("data:image/"):
            header, encoded = image_source.split(",", 1)
            return "image.jpg", base64.b64decode(encoded)

        # 3. Direct File System Path
        if os.path.exists(image_source) and os.path.isfile(image_source):
            with open(image_source, "rb") as f:
                return os.path.basename(image_source), f.read()

        # 4. Open WebUI File ID or Upload Search (matches 3ae432b7-27a5-4eb1-b845-da9c10443351)
        search_dirs = [
            "/app/backend/data/uploads",
            "/app/backend/data/cache",
            "/app/backend/data",
            "/tmp",
            "./data",
        ]
        clean_id = os.path.basename(image_source)
        for base_dir in search_dirs:
            if not os.path.exists(base_dir):
                continue
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    if clean_id in fname or fname.startswith(clean_id):
                        full_path = os.path.join(root, fname)
                        with open(full_path, "rb") as f:
                            return fname, f.read()

        # 5. Open WebUI internal API fetch by File ID
        for port in [8080, 8000]:
            try:
                r_api = requests.get(
                    f"http://127.0.0.1:{port}/api/v1/files/{clean_id}/content",
                    timeout=5,
                )
                if r_api.status_code == 200 and len(r_api.content) > 0:
                    return f"{clean_id}.jpg", r_api.content
            except Exception:
                pass

        raise FileNotFoundError(
            f"Could not locate image file or ID '{image_source}' on server or disk."
        )

    def classify_image(self, image_source: str) -> str:
        """
        Classify an image using the deployed computer vision model.
        :param image_source: Local file path, URL (http/https), base64 data URI, or Open WebUI uploaded file ID.
        :return: JSON string containing predicted class, confidence score, top predictions, latency, and model version.
        """
        try:
            filename, file_bytes = self._find_image_bytes(image_source)
            files = {"image": (filename, io.BytesIO(file_bytes), "image/jpeg")}
            result = self._request("POST", "/api/v1/predict", files=files)
            self._log_langfuse("classify_image", {"image_source": image_source}, result)
            return result
        except Exception as e:
            err = f"Error classifying image: {str(e)}"
            self._log_langfuse("classify_image", {"image_source": image_source}, err)
            return err

    def get_model_info(self) -> str:
        """
        Get information about the currently deployed computer vision model.
        :return: JSON string with model metadata, version, classes, and metrics.
        """
        try:
            result = self._request("GET", "/api/v1/model")
            self._log_langfuse("get_model_info", {}, result)
            return result
        except Exception as e:
            err = f"Error retrieving model info: {str(e)}"
            self._log_langfuse("get_model_info", {}, err)
            return err

    def get_prediction_history(self, limit: int = 5) -> str:
        """
        Get the most recent N predictions from the PostgreSQL database.
        :param limit: Number of recent predictions to fetch (1-50).
        :return: JSON string containing list of recent predictions.
        """
        try:
            result = self._request(
                "GET",
                "/api/v1/predictions",
                params={"limit": limit, "offset": 0},
            )
            self._log_langfuse("get_prediction_history", {"limit": limit}, result)
            return result
        except Exception as e:
            err = f"Error retrieving prediction history: {str(e)}"
            self._log_langfuse("get_prediction_history", {"limit": limit}, err)
            return err

    def get_prediction_by_id(self, prediction_id: int) -> str:
        """
        Get details of a specific prediction by its numeric ID.
        :param prediction_id: Numeric ID of the prediction record.
        :return: JSON string containing prediction details.
        """
        try:
            result = self._request("GET", f"/api/v1/predictions/{prediction_id}")
            self._log_langfuse("get_prediction_by_id", {"prediction_id": prediction_id}, result)
            return result
        except Exception as e:
            err = f"Error retrieving prediction #{prediction_id}: {str(e)}"
            self._log_langfuse("get_prediction_by_id", {"prediction_id": prediction_id}, err)
            return err

    def get_prediction_statistics(self) -> str:
        """
        Get aggregate prediction statistics including total predictions and class distributions.
        :return: JSON string containing prediction statistics.
        """
        try:
            result = self._request("GET", "/api/v1/stats")
            self._log_langfuse("get_prediction_statistics", {}, result)
            return result
        except Exception as e:
            err = f"Error retrieving prediction statistics: {str(e)}"
            self._log_langfuse("get_prediction_statistics", {}, err)
            return err
