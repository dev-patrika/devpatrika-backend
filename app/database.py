from sqlmodel import create_engine, Session, SQLModel
from app.config import settings

# Configure SQLAlchemy engine for Neon Postgres (pooled connection)
# pool_pre_ping: checks connection liveness before each use (important for serverless Postgres)
# pool_size: max persistent connections in the pool
# max_overflow: extra connections allowed above pool_size under burst load
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10
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
