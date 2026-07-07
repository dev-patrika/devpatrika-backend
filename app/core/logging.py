import logging
import sys

def setup_logging():
    # Setup custom log formatting
    log_format = (
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Optional: reduce verbosity of internal libraries
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    logger = logging.getLogger("dev-patrika")
    logger.info("Logging successfully initialized")
    return logger

logger = setup_logging()
