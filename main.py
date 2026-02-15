"""

Este bot usa:
- Busca híbrida (vetorial + BM25)
- LLM local via Ollama
- Persistência em ChromaDB
- Interface via Telegram
"""
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import Command

from config import TELEGRAM_TOKEN, LLM_MODEL, QUERY_TIMEOUT, STORAGE_DIR
from core import InstitutionalHybridBot
from bot import BotHandlers
from utils.logger import logger


def print_startup_banner():
    print(f"\n{'='*60}")
    print(f"🚀 SECRETÁRIO BOT ONLINE")
    print(f"{'='*60}")
    print(f"📚 Modelo LLM: {LLM_MODEL}") # Atualizado aqui
    print(f"⏱️  Timeout: {QUERY_TIMEOUT}s")


async def start_bot_service():
    """Inicia o serviço do bot."""
    # Inicializar bot e dispatcher
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()
    
    # Inicializar engine
    try:
        logger.info("🚀 Inicializando engine...")
        engine_instance = InstitutionalHybridBot()
        logger.info("✅ Engine inicializada com sucesso!")
    except Exception as e:
        logger.error(f"❌ Falha ao inicializar: {e}", exc_info=True)
        raise
    
    # Configurar handlers
    handlers = BotHandlers(engine_instance)
    dp.message.register(handlers.cmd_start, Command("start"))
    dp.message.register(handlers.cmd_debug_context, Command("contexto"))
    dp.message.register(handlers.handle_query)
    
    # Exibir banner
    print_startup_banner()
    
    # Iniciar polling
    await dp.start_polling(bot)


def main():
    """Função principal."""
    try:
        asyncio.run(start_bot_service())
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 Bot encerrado com sucesso.")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()