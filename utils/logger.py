"""
Configuração centralizada de logging.
"""
import logging
from config.settings import LOG_LEVEL, LOG_FORMAT

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configura e retorna um logger.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo)
    
    Returns:
        Logger configurado
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    return logging.getLogger(name)

# Logger global
logger = setup_logger("chatbot")