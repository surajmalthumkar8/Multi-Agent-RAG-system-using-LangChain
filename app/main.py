"""
Multi-Agent RAG System - Main Application
==========================================

FastAPI application entry point for the Multi-Agent RAG system.

This module:
- Creates the FastAPI application
- Configures middleware and CORS
- Includes API routes
- Sets up logging
- Provides startup/shutdown hooks

RUNNING THE APP:
    Development: uvicorn app.main:app --reload
    Production:  uvicorn app.main:app --host 0.0.0.0 --port 8000

API DOCUMENTATION:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.config import get_settings
from app import __version__

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> None:
    """
    Configure logging for the application.

    Sets up structured logging with appropriate levels and formatting.
    """
    settings = get_settings()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize logging, ensure directories exist
    - Shutdown: Cleanup resources

    Args:
        app: FastAPI application instance
    """
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)

    settings = get_settings()
    settings.ensure_directories()

    logger.info(f"Starting Multi-Agent RAG System v{__version__}")
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"Documents path: {settings.documents_path}")
    logger.info(f"FAISS index path: {settings.faiss_index_path}")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Multi-Agent RAG System")


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    settings = get_settings()

    app = FastAPI(
        title="Multi-Agent RAG System",
        description=(
            "A production-grade Retrieval-Augmented Generation system "
            "using multiple specialized agents for query routing, "
            "document retrieval, reasoning, and action execution."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to API documentation."""
        return RedirectResponse(url="/docs")

    return app


# Create the application instance
app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower(),
    )
