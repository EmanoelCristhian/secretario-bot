"""
Templates de mensagens do bot.
"""
from config import LLM_MODEL, QUERY_TIMEOUT


class BotMessages:
    """Mensagens padronizadas do bot."""
    
    @staticmethod
    def welcome_message() -> str:
        return (
            f"🤖 *Assistente Institucional Ativo*\n\n"
            f"📚 Base de conhecimento: Regulamentos acadêmicos\n"
            f"🧠 Modelo: {LLM_MODEL}\n" # Atualizado aqui
            f"🔍 Busca: Híbrida (Vetorial + BM25)\n\n"
            f"💡 Envie sua pergunta!"
        )
    
    @staticmethod
    def greeting_response() -> str:
        """Resposta a saudações simples."""
        return (
            f"👋 Olá! Seja bem-vindo(a)!\n\n"
            f"Sou o assistente institucional e posso ajudar com informações sobre "
            f"regulamentos acadêmicos, TCC, matrículas e muito mais.\n\n"
            f"💡 Como posso ajudar você hoje?"
        )
    
    @staticmethod
    def greeting_with_query_intro() -> str:
        """Introdução quando há saudação + pergunta."""
        return (
            f"👋 Olá! Vou processar sua pergunta...\n\n"
        )
    
    @staticmethod
    def processing_message() -> str:
        """Mensagem durante processamento."""
        return "⏳ Processando sua pergunta..."
    
    @staticmethod
    def timeout_message() -> str:
        """Mensagem de timeout."""
        return (
            f"⏰ A consulta excedeu o tempo limite de {QUERY_TIMEOUT}s.\n"
            f"💡 Tente uma pergunta mais específica."
        )
    
    @staticmethod
    def error_message(error: str) -> str:
        """Mensagem de erro."""
        return (
            f"❌ Erro ao processar sua solicitação.\n\n"
            f"🔧 Detalhes: {error[:150]}\n\n"
            f"💡 Tente novamente ou contate o administrador."
        )
    
    @staticmethod
    def small_talk_response() -> str:
        """Resposta para mensagens fora do escopo acadêmico."""
        return (
            "😊 Entendido! Só consigo ajudar com assuntos da FCT/UFPA — "
            "disciplinas, ementas, regulamentos, TCC, estágio e informações institucionais.\n\n"
            "Tem alguma dúvida sobre o seu curso? É só perguntar!"
        )

    @staticmethod
    def not_found_response(query: str) -> str:
        """Resposta quando não encontra informação nos documentos."""
        return (
            f"Não encontrei informações sobre *\"{query}\"* nos documentos da FCT.\n\n"
            "💡 Tente reformular a pergunta ou pergunte sobre:\n"
            "• Disciplinas e ementas dos cursos\n"
            "• Regulamento de TCC ou Estágio\n"
            "• Grade curricular por bloco\n"
            "• Atividades complementares"
        )

    @staticmethod
    def truncation_warning() -> str:
        """Aviso de resposta truncada."""
        return "..."