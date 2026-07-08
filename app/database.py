from sqlmodel import create_engine, Session, SQLModel
from app.config import settings

# SQLite requires connect_args={"check_same_thread": False} for multi-thread requests
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args=connect_args
)

def init_db():
    """Create database tables if they do not exist"""
    # Import models here to ensure metadata registration
    from app import models
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency injection session generator"""
    with Session(engine) as session:
        yield session
