from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("dev-patrika")

class DevPatrikaException(Exception):
    """Base exception for all Dev Patrika custom exceptions"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DatabaseException(DevPatrikaException):
    """Raised when database operations fail"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

class IngestionException(DevPatrikaException):
    """Raised when parsing or fetching external feeds fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=502)

class AIProcessingException(DevPatrikaException):
    """Raised when LangChain prompt execution fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=502)

async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, DevPatrikaException):
        logger.error(f"Custom Error handling {request.url}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "detail": exc.message}
        )
    
    logger.exception(f"Unhandled system error handling {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "An unexpected server error occurred."}
    )
