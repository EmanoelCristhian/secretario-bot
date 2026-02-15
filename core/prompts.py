"""
Templates de prompts para o LLM.
"""
from typing import List
from llama_index.core.schema import NodeWithScore


class PromptTemplates:
    """Templates de prompts contextualizados."""
    
    @staticmethod
    def build_system_message() -> str:
        """
        Mensagem de sistema para configurar o comportamento do LLM.
        
        Returns:
            Mensagem de sistema
        """
        return """Você é um assistente acadêmico especializado em informações sobre o curso de Engenharia da Computação.

Suas responsabilidades:
- Fornecer informações precisas sobre disciplinas, TCC, matrículas, regulamentos
- Citar os documentos oficiais quando disponível
- Admitir quando não tem informação ao invés de especular
- Ser claro, objetivo e educado

Sempre baseie suas respostas nos documentos fornecidos."""

class ResponseValidator:
    """Valida e melhora respostas do LLM."""
    
    @staticmethod
    def validate_response(response: str, query: str) -> str:
        """
        Valida e melhora a resposta.
        
        Args:
            response: Resposta do LLM
            query: Pergunta original
            
        Returns:
            Resposta validada
        """
        response = response.strip()
        
        # Remover prefixos comuns indesejados
        prefixes_to_remove = [
            "RESPOSTA:",
            "Resposta:",
            "Com base nos documentos,",
            "De acordo com os documentos,",
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Se resposta muito curta, adicionar contexto
        if len(response) < 20:
            response = f"Desculpe, não encontrei informações suficientes nos documentos para responder: '{query}'"
        
        return response
    
    @staticmethod
    def detect_hallucination_indicators(response: str) -> bool:
        """
        Detecta indicadores de alucinação.
        
        Args:
            response: Resposta a verificar
            
        Returns:
            True se há indicadores de alucinação
        """
        hallucination_phrases = [
            "eu acho que",
            "provavelmente",
            "deve ser",
            "possivelmente",
            "na minha opinião",
            "acredito que",
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in hallucination_phrases)