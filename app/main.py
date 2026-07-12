from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.routers import all_routers
from app.core.exceptions import global_exception_handler
from app.services.ingestion.scheduler import start_scheduler
import logging

# Initialize logging configuration
logger = logging.getLogger("dev-patrika")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the database tables
    logger.info("Initializing SQLite database tables...")
    try:
        init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failure: {str(e)}")
        raise e
    
    # Start the ingestion scheduler in the background
    start_scheduler()
    
    yield
    # Shutdown actions (none required for SQLite)

app = FastAPI(
    title="Dev Patrika API",
    description="An AI-Powered Developer Intelligence Platform API Backend",
    version="0.1.0-alpha",
    lifespan=lifespan
)

# Configure CORS Middleware (allowing frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=settings.JWT_SECRET)

# Register Global Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)

# Register API Routers dynamically
for router, prefix in all_routers:
    app.include_router(router, prefix=prefix)
