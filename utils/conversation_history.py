"""
Gerenciamento de histórico de conversa por usuário.
"""
from collections import deque
from config import HISTORY_MAX_TURNS


class ConversationHistory:
    """
    Mantém um buffer circular de turnos (pergunta + resposta) por usuário.

    Cada turno é um dict {"role": "user"|"assistant", "content": str}.
    O buffer é limitado a HISTORY_MAX_TURNS pares para não inflar o prompt.
    """

    def __init__(self):
        self._histories: dict[int, deque] = {}

    def add_turn(self, user_id: int, question: str, answer: str) -> None:
        if user_id not in self._histories:
            self._histories[user_id] = deque(maxlen=HISTORY_MAX_TURNS * 2)
        buf = self._histories[user_id]
        buf.append({"role": "user", "content": question})
        buf.append({"role": "assistant", "content": answer})

    def get_prompt_block(self, user_id: int) -> str:
        """
        Retorna o histórico formatado para ser inserido no prompt.
        Retorna string vazia se não houver histórico.
        """
        buf = self._histories.get(user_id)
        if not buf:
            return ""

        lines = ["### HISTÓRICO DA CONVERSA (contexto anterior):"]
        for msg in buf:
            prefix = "Usuário" if msg["role"] == "user" else "Assistente"
            lines.append(f"{prefix}: {msg['content']}")
        lines.append("")
        return "\n".join(lines)

    def clear(self, user_id: int) -> None:
        self._histories.pop(user_id, None)
