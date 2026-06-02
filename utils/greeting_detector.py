"""
Detector de saudações e cumprimentos.
"""
import re
from typing import Tuple


class GreetingDetector:
    """Detecta se uma mensagem é uma saudação."""
    
    # Padrões de saudações em português (inclui gírias informais)
    GREETING_PATTERNS = [
        r'\b(oi|olá|ola|hey|opa|e aí|eae|eai|aoba|aob)\b',
        r'\b(bom dia|boa tarde|boa noite)\b',
        r'\b(tudo bem|tudo bom|como vai|td bem|blz|tmj)\b',
        r'\b(salve|fala|coé|coe|iae|iáe|e ai|e aí)\b',
        r'\b(ola+|oii+|oin+|oie|oioi|alô|alo)\b',
        r'^(oi+|olá+|hey+|opa+|salve+)$',  # Apenas saudação
        r'^/start$',  # Comando start
    ]

    # Termos que indicam conversa fora do escopo acadêmico
    OFF_TOPIC_INDICATORS = [
        r'\b(é um cumprimento|é uma gíria|é uma expressão|significa|quer dizer)\b',
        r'\b(me conta|conta mais|fala sobre|explica o que é)\b',
        r'^(ok|okay|certo|entendi|legal|vlw|valeu|obrigad[ao]|kkk+|rs+|haha+)$',
        r'^\w{1,15}$',  # Palavra isolada curta sem contexto acadêmico
    ]
    
    # Padrões que indicam pergunta (não é só saudação)
    QUESTION_PATTERNS = [
        r'\?',  # Tem ponto de interrogação
        r'\b(o que|como|quando|onde|por que|porque|qual|quais|quantos|quantas)\b',
        r'\b(pode|poderia|consegue|sabe|tem como)\b',
        r'\b(me (diga|fala|mostra|explica|ajuda))\b',
        r'\b(quero saber|gostaria de saber|preciso saber)\b',
    ]
    
    # Termos que indicam contexto acadêmico (mesmo sem "?" explícito)
    ACADEMIC_PATTERNS = [
        r'\b(disciplina|matéria|ementa|bloco|crédito|carga horária)\b',
        r'\b(tcc|estágio|monografia|defesa|banca)\b',
        r'\b(engenharia|computação|telecomunicações|fct|ufpa)\b',
        r'\b(regulamento|norma|resolução|portaria|prazo)\b',
        r'\b(matrícula|aproveitamento|equivalência|grade|currículo)\b',
        r'\b(professor|coordenador|colegiado|atividade|complementar)\b',
    ]

    def __init__(self):
        """Compila os padrões regex."""
        self.greeting_regex = re.compile(
            '|'.join(self.GREETING_PATTERNS),
            re.IGNORECASE
        )
        self.question_regex = re.compile(
            '|'.join(self.QUESTION_PATTERNS),
            re.IGNORECASE
        )
        self.off_topic_regex = re.compile(
            '|'.join(self.OFF_TOPIC_INDICATORS),
            re.IGNORECASE
        )
        self.academic_regex = re.compile(
            '|'.join(self.ACADEMIC_PATTERNS),
            re.IGNORECASE
        )
    
    def is_greeting(self, text: str) -> bool:
        """
        Verifica se o texto contém uma saudação.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True se contém saudação
        """
        return bool(self.greeting_regex.search(text))
    
    def is_question(self, text: str) -> bool:
        """
        Verifica se o texto contém uma pergunta.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True se contém pergunta
        """
        return bool(self.question_regex.search(text))
    
    def is_pure_greeting(self, text: str) -> bool:
        """
        Verifica se é APENAS uma saudação (sem pergunta).
        
        Args:
            text: Texto a verificar
            
        Returns:
            True se é saudação pura (sem pergunta)
        """
        text_clean = text.strip().lower()
        
        # Se é muito curto (até 20 caracteres) e tem saudação
        if len(text_clean) <= 20 and self.is_greeting(text_clean):
            return not self.is_question(text_clean)
        
        # Se tem saudação mas também tem pergunta, não é pura
        if self.is_greeting(text_clean) and self.is_question(text_clean):
            return False
        
        # Se tem saudação e é curto, é pura
        if self.is_greeting(text_clean) and len(text_clean.split()) <= 5:
            return True
        
        return False
    
    def classify_message(self, text: str) -> Tuple[bool, bool]:
        """
        Classifica a mensagem.
        
        Args:
            text: Texto a classificar
            
        Returns:
            Tupla (is_greeting, has_question)
            - (True, False): Apenas saudação
            - (True, True): Saudação + pergunta
            - (False, True): Apenas pergunta
            - (False, False): Outro tipo de mensagem
        """
        is_greet = self.is_greeting(text)
        has_quest = self.is_question(text)
        
        return (is_greet, has_quest)

    def is_small_talk(self, text: str) -> bool:
        """
        Detecta conversa casual fora do escopo acadêmico.

        Retorna True quando a mensagem é curta, não contém termos acadêmicos
        e não é uma pergunta com "?". Exemplos: "aoba é um cumprimento",
        "ok", "valeu", "kkk", palavras isoladas sem contexto.
        """
        text_clean = text.strip()

        # Se tem termos acadêmicos, não é small talk
        if self.academic_regex.search(text_clean):
            return False

        # Se tem "?" é uma pergunta — deixar para o RAG tentar
        if '?' in text_clean:
            return False

        # Palavra/frase muito curta sem interrogação e sem contexto acadêmico
        if len(text_clean) <= 50 and not self.is_question(text_clean):
            return True

        return False