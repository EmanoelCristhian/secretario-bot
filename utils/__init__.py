"""
Utilitários do sistema.
"""
from .logger import setup_logger, logger
from .greeting_detector import GreetingDetector

__all__ = ["setup_logger", "logger", "GreetingDetector"]