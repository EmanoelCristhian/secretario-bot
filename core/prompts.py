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
        return """Você é um assistente acadêmico especializado em informações sobre os cursos de Engenharia da Computação e Engenharia de Telecomunicações da UFPA (Faculdade de Engenharia da Computação e Telecomunicações — FCT).

Suas responsabilidades:
- Fornecer informações precisas sobre disciplinas, blocos curriculares, TCC, estágio supervisionado, matrículas e regulamentos de ambos os cursos
- Citar os documentos oficiais quando disponível
- Admitir quando não tem informação ao invés de especular
- Ser claro, objetivo e educado

Sempre baseie suas respostas exclusivamente nos documentos fornecidos."""

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

        # Detectar respostas em inglês geradas pelo LlamaIndex quando não há contexto.
        # Exemplos reais: "The provided context does not contain information about X"
        #                 "I cannot find information about X in the provided context"
        english_not_found_patterns = [
            "the provided context does not",
            "the context does not contain",
            "i cannot find information",
            "i don't have information",
            "there is no information",
            "no information available",
            "the documents do not contain",
            "i was unable to find",
        ]
        response_lower = response.lower()
        if any(pat in response_lower for pat in english_not_found_patterns):
            response = (
                f"Não encontrei informações sobre \"{query}\" nos documentos da FCT.\n\n"
                "💡 Tente reformular a pergunta ou pergunte sobre disciplinas, "
                "ementas, regulamentos de TCC, estágio ou grade curricular."
            )

        # Se resposta muito curta, usar fallback em português
        if len(response) < 20:
            response = (
                f"Não encontrei informações suficientes nos documentos para responder: \"{query}\""
            )

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