import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, history, model_info, predictions, stats
from app.config import get_settings
from app.database import init_db

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv-agent-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database tables ensured. Model version=%s", settings.MODEL_VERSION)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.MODEL_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(history.router)
app.include_router(stats.router)
app.include_router(model_info.router)


# --- Global error handling: never leak tracebacks to clients (section 33) ---


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": None},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": None},
    )

