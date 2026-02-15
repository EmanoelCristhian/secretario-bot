"""
Handlers do bot Telegram.
"""
import asyncio
from aiogram import types
from aiogram.filters import Command

from config import QUERY_TIMEOUT, MAX_RESPONSE_LENGTH
from bot.messages import BotMessages
from utils.logger import logger
from utils import GreetingDetector


class BotHandlers:
    """Gerencia os handlers do bot Telegram."""
    
    def __init__(self, engine_instance):
        """
        Inicializa os handlers.
        
        Args:
            engine_instance: Instância do motor de busca
        """
        self.engine = engine_instance
        self.messages = BotMessages()
        self.greeting_detector = GreetingDetector()
        self.users_started = set()

    async def cmd_start(self, message: types.Message):
        """Handler do comando /start."""
        user_id = message.from_user.id
        self.users_started.add(user_id)
        
        await message.answer(
            self.messages.welcome_message(),
            parse_mode="Markdown"
        )

    async def cmd_debug_context(self, message: types.Message):
        """
        Comando de debug: mostra contexto recuperado sem gerar resposta.
        
        Uso: /contexto <sua pergunta>
        """
        user_id = message.from_user.id
        
        # Extrair texto após o comando
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.answer(
                "❓ Uso: /contexto <sua pergunta>\n\n"
                "Exemplo: /contexto quais são as disciplinas do curso?"
            )
            return
        
        query = parts[1]
        logger.info(f"🔍 Debug contexto solicitado por {user_id}: '{query}'")
        
        processing_msg = await message.answer("🔍 Buscando contexto...")
        
        try:
            context = await asyncio.to_thread(
                self.engine.get_context_for_query,
                query
            )
            
            await processing_msg.delete()
            
            # Truncar se muito longo
            if len(context) > 3500:
                context = context[:3497] + "..."
            
            await message.answer(
                f"📚 *Contexto Recuperado:*\n\n{context}",
                parse_mode="Markdown"
            )
            
        except Exception as e:
            logger.error(f"❌ Erro no debug: {e}")
            await processing_msg.edit_text(f"❌ Erro: {str(e)[:150]}")

    async def handle_query(self, message: types.Message):
        """
        Handler de queries do usuário.
        
        Detecta se é:
        1. Apenas saudação → responde com greeting_response
        2. Saudação + pergunta → responde saudação E processa pergunta
        3. Apenas pergunta → processa pergunta normalmente
        
        Args:
            message: Mensagem do Telegram
        """
        user_id = message.from_user.id
        user_text = message.text
        
        logger.info(f"📨 Mensagem de {user_id}: '{user_text}'")
        
        # Classificar a mensagem
        is_greeting, has_question = self.greeting_detector.classify_message(user_text)
        
        # Caso 1: Apenas saudação (sem pergunta)
        if is_greeting and not has_question:
            logger.info(f"👋 Saudação detectada de {user_id} (sem pergunta)")
            await message.answer(self.messages.greeting_response())
            # Marcar que usuário já foi "iniciado"
            self.users_started.add(user_id)
            return
        
        # Caso 2: Saudação + pergunta
        if is_greeting and has_question:
            logger.info(f"👋 Saudação + pergunta detectada de {user_id}")
            # Enviar saudação primeiro (se ainda não usou /start)
            if user_id not in self.users_started:
                await message.answer(self.messages.greeting_with_query_intro())
                self.users_started.add(user_id)
            # Continua para processar a pergunta abaixo
        
        # Caso 3: Apenas pergunta (ou saudação + pergunta)
        # Processar a query normalmente
        await self._process_query(message, user_text, user_id)

    async def _process_query(self, message: types.Message, user_text: str, user_id: int):
        """
        Processa uma query do usuário.
        
        Args:
            message: Mensagem do Telegram
            user_text: Texto da mensagem
            user_id: ID do usuário
        """
        # Feedback imediato
        processing_msg = await message.answer(
            self.messages.processing_message()
        )
        
        try:
            # Processar query com timeout
            logger.info("🔄 Iniciando processamento...")
            response = await asyncio.wait_for(
                asyncio.to_thread(self.engine.query, user_text),
                timeout=QUERY_TIMEOUT
            )
            
            logger.info("✅ Resposta obtida, enviando...")
            
            # Remover mensagem de processamento
            await processing_msg.delete()
            
            # Preparar resposta
            response_text = self._prepare_response(str(response), user_id)
            
            # Enviar resposta
            await message.answer(response_text)
            logger.info(f"✅ Resposta enviada para {user_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout ao processar query de {user_id}")
            await processing_msg.edit_text(self.messages.timeout_message())
            
        except Exception as e:
            logger.error(
                f"❌ Erro ao processar query de {user_id}: {e}",
                exc_info=True
            )
            await processing_msg.edit_text(
                self.messages.error_message(str(e))
            )

    def _prepare_response(self, response_text: str, user_id: int) -> str:
        """
        Prepara resposta respeitando limites do Telegram.
        
        Args:
            response_text: Texto da resposta
            user_id: ID do usuário (para logs)
            
        Returns:
            Resposta formatada
        """
        if len(response_text) > MAX_RESPONSE_LENGTH:
            response_text = (
                response_text[:MAX_RESPONSE_LENGTH - 3] +
                self.messages.truncation_warning()
            )
            logger.warning(f"⚠️ Resposta truncada para {user_id}")
        
        return response_text
