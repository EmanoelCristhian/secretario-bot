"""
Módulo de avaliação quantitativa utilizando o framework RAGAS.
"""
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from core.engine import InstitutionalHybridBot
from utils.logger import logger
from config import GOOGLE_API_KEY, JUDGE_MODEL


API_CONFIG = RunConfig(
    max_workers=1,
    max_retries=15,
    max_wait=60,
)


class RAGEvaluator:
    """Validação estatística do motor RAG com framework RAGAS."""

    def __init__(self):
        logger.info("🧪 Inicializando Avaliador RAGAS...")

        self.judge_llm = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            max_retries=3,
        )

        logger.info("⚙️ Carregando Embeddings locais para o Juiz...")
        self.judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        self.metrics = [
            Faithfulness(),     # Resposta é fiel ao contexto recuperado?
            AnswerRelevancy(),  # Resposta é relevante para a pergunta?
            ContextPrecision(), # Contexto recuperado contém a ground-truth?
            ContextRecall(),    # Ground-truth está coberta pelo contexto?
        ]

    def run_evaluation(self, test_set: list, retrieval_mode: str = "hybrid") -> pd.DataFrame:
        """
        Executa o test set em um modo de retrieval e retorna as métricas.

        Args:
            test_set: Lista de dicts com user_input e reference.
            retrieval_mode: "hybrid", "vector" ou "bm25".
        """
        logger.info(f"▶️ Iniciando avaliação ({retrieval_mode}) com {len(test_set)} questões...")

        bot = InstitutionalHybridBot(retrieval_mode=retrieval_mode)

        user_inputs, responses, retrieved_contexts, references = [], [], [], []

        for item in test_set:
            q = item["user_input"]
            logger.info(f"  Perguntando: {q[:80]}")
            try:
                response = bot.query_engine.query(q)
                ans = str(response)
                ctx = [n.node.get_content() for n in response.source_nodes]
            except Exception as e:
                logger.error(f"  Erro em '{q}': {e}")
                ans = "Erro na geração"
                ctx = ["Erro"]

            user_inputs.append(q)
            responses.append(ans)
            retrieved_contexts.append(ctx)
            references.append(item["reference"])

        dataset = Dataset.from_dict({
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        })

        logger.info("🧠 Calculando métricas RAGAS...")
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            embeddings=self.judge_embeddings,
            run_config=API_CONFIG,
        )

        df = result.to_pandas()
        df.insert(0, "retrieval_mode", retrieval_mode)
        return df

    def run_comparative_evaluation(self, test_set: list) -> pd.DataFrame:
        """
        Avalia os três modos de retrieval e retorna um DataFrame consolidado.
        Gera também um resumo com as médias de cada métrica por modo.
        """
        frames = []
        for mode in ("hybrid", "vector", "bm25"):
            logger.info(f"\n{'='*60}\n🔬 Avaliando modo: {mode}\n{'='*60}")
            df = self.run_evaluation(test_set, retrieval_mode=mode)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        return combined


# ---------------------------------------------------------------------------
# Test set — Engenharia da Computação (níveis 1-3) + Telecomunicações
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ==========================================
    # NÍVEL 1 — PERGUNTAS DIRETAS (Factoides)
    # ==========================================
    {
        "user_input": "Qual é a carga horária das disciplinas de TCC I e TCC II?",
        "reference": "O desenvolvimento do TCC se dá em duas etapas (TCC I e TCC II), cujas cargas horárias são de 120 (cento e vinte) horas cada uma."
    },
    {
        "user_input": "Qual é o número mínimo de páginas exigido para o documento final do TCC?",
        "reference": "O número mínimo de páginas do TCC é de 35 (trinta e cinco), sem contar com apêndices e anexos."
    },
    {
        "user_input": "Qual o tempo máximo de duração da defesa de TCC e como esse tempo é dividido?",
        "reference": "A duração da defesa é de no máximo 60 (sessenta) minutos, distribuídos em 30 minutos para a apresentação do discente e o restante para arguição da banca e manifestação da audiência."
    },
    {
        "user_input": "Quantas horas no mínimo o aluno precisa completar no Estágio Supervisionado?",
        "reference": "O discente deverá completar um mínimo de 390 horas de Estágio Supervisionado."
    },
    {
        "user_input": "Quais são as opções de carga horária e créditos para as disciplinas de Estágio Supervisionado?",
        "reference": "As disciplinas são ofertadas com 100 horas (1 crédito), 200 horas (2 créditos) e 390 horas (4 créditos)."
    },

    # ==========================================
    # NÍVEL 2 — PERGUNTAS PROCEDIMENTAIS
    # ==========================================
    {
        "user_input": "Qual é o passo a passo para solicitar a matrícula em TCC no sistema SAGITTA?",
        "reference": "No início do semestre, o aluno deve acessar o SAGITTA > Nova Chamada > ITEC > FACULDADE DE ENGENHARIA DA COMP. E TELECOM > Trabalho de Conclusão de Curso, e anexar o formulário específico."
    },
    {
        "user_input": "Após a defesa do TCC, qual o procedimento para entregar a versão final digital?",
        "reference": "O aluno deve enviar um e-mail para engcomp@ufpa.br e hewerton@ufpa.br, com cópia para o orientador, que deve confirmar que aquele documento é a versão final. Além disso, deve disponibilizar uma cópia em PDF para cada membro da banca."
    },
    {
        "user_input": "Qual o prazo para o aluno apresentar a versão corrigida do TCC após a defesa?",
        "reference": "Após a defesa, o discente terá até 10 (dez) dias para apresentar a versão corrigida, atendendo por completo às observações da banca examinadora."
    },
    {
        "user_input": "Qual é o prazo para a entrega do Relatório Final de Estágio Supervisionado?",
        "reference": "O discente deve submeter o Relatório Final ao colegiado no prazo máximo de sete dias antes do final das aulas do semestre letivo, contendo a assinatura e nota do coordenador de estágio."
    },

    # ==========================================
    # NÍVEL 3 — RACIOCÍNIO LÓGICO E CONDICIONAIS
    # ==========================================
    {
        "user_input": "Se o trabalho de TCC for desenvolvido dentro de um projeto de pesquisa em equipe de 3 pessoas, podemos apresentar o TCC em trio?",
        "reference": "Não. Os TCCs devem ser realizados individualmente, não sendo permitidos trabalhos em duplas ou trios. O orientador deve dividir as tarefas para que cada discente apresente um TCC sólido e distinto."
    },
    {
        "user_input": "O que é exigido como pré-requisito para que um aluno possa se matricular na disciplina de Estágio Supervisionado de 390 horas?",
        "reference": "A matrícula só será efetivada se o aluno já tiver obtido aprovação em todas as disciplinas obrigatórias do primeiro, segundo e quinto blocos, além de ter um Plano de Estágio aprovado pelo colegiado do curso."
    },
    {
        "user_input": "Como deve ser composta a banca examinadora do TCC caso o aluno tenha apenas um orientador, sem co-orientador?",
        "reference": "Sugere-se que a banca seja composta por três membros. Exige-se que haja um mínimo de dois professores do quadro efetivo da UFPA na banca."
    },
    {
        "user_input": "Eu reprovei em 3 disciplinas dos primeiros semestres. Posso me matricular em TCC I?",
        "reference": "Não. O discente não pode ser matriculado em TCC I caso tenha mais do que duas dependências em disciplinas de blocos anteriores ao sétimo, além da obrigação de já ter sido aprovado no sexto bloco."
    },
    {
        "user_input": "Comecei a estagiar no ano passado e esqueci de matricular. Posso aproveitar todas as minhas horas antigas no Plano de Estágio deste semestre?",
        "reference": "Não totalmente. O discente pode apresentar um Plano de Estágio que inclua atividades de até quatro meses antes do início da matrícula. Atividades anteriores a esses quatro meses não poderão ser contabilizadas retroativamente."
    },
    {
        "user_input": "Se eu reprovar no TCC I, posso matricular no TCC II ao mesmo tempo no próximo semestre para adiantar?",
        "reference": "Não. O discente só poderá ser matriculado em TCC II após ser aprovado em TCC I."
    },

    # ==========================================
    # TELECOMUNICAÇÕES — NÍVEL 1 (Factoides)
    # ==========================================
    {
        "user_input": "Quais são as disciplinas obrigatórias do primeiro bloco do curso de Engenharia de Telecomunicações?",
        "reference": "O primeiro bloco do curso de Engenharia de Telecomunicações inclui disciplinas como Cálculo Diferencial e Integral I, Geometria Analítica, Introdução à Engenharia de Telecomunicações, Algoritmos e Programação, e Desenho Técnico."
    },
    {
        "user_input": "Qual é a carga horária total do curso de Engenharia de Telecomunicações da UFPA?",
        "reference": "A carga horária total do curso de Engenharia de Telecomunicações da UFPA é de 3.840 horas."
    },
    {
        "user_input": "Quais disciplinas de Sistemas de Comunicação estão presentes na grade curricular de Telecomunicações?",
        "reference": "A grade curricular de Telecomunicações inclui disciplinas como Sistemas de Comunicações I, Sistemas de Comunicações II, Comunicações Ópticas e Comunicações Móveis."
    },

    # ==========================================
    # TELECOMUNICAÇÕES — NÍVEL 2 (Procedimentais)
    # ==========================================
    {
        "user_input": "Quais são os requisitos para integralização do curso de Engenharia de Telecomunicações?",
        "reference": "Para integralizar o curso de Engenharia de Telecomunicações, o aluno deve completar todas as disciplinas obrigatórias dos dez blocos curriculares, as atividades complementares exigidas, o estágio supervisionado e o Trabalho de Conclusão de Curso."
    },
    {
        "user_input": "Como funciona a estrutura de blocos do curso de Telecomunicações e quantos blocos existem?",
        "reference": "O curso de Engenharia de Telecomunicações é organizado em dez blocos curriculares sequenciais, onde cada bloco agrupa disciplinas de um semestre letivo e serve de pré-requisito para o bloco seguinte."
    },
]


if __name__ == "__main__":
    import sys

    evaluator = RAGEvaluator()

    # Modo padrão: avaliação comparativa completa (3 modos)
    # Para rodar só um modo: python -m core.evaluator hybrid
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "comparative"

    if mode_arg == "comparative":
        combined = evaluator.run_comparative_evaluation(TEST_CASES)

        print("\n" + "=" * 80)
        print("📊 AVALIAÇÃO COMPARATIVA — RAGAS")
        print("=" * 80)

        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        summary = (
            combined.groupby("retrieval_mode")[metric_cols]
            .mean()
            .round(4)
            .sort_values("context_precision", ascending=False)
        )
        print(summary.to_string())

        output_path = "resultados_tcc_comparativo.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Resultado completo salvo em '{output_path}'")

    else:
        df = evaluator.run_evaluation(TEST_CASES, retrieval_mode=mode_arg)

        print("\n" + "=" * 80)
        print(f"📊 RESULTADOS RAGAS — modo: {mode_arg}")
        print("=" * 80)

        cols = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print(df[cols].to_string(index=False))

        output_path = f"resultados_tcc_{mode_arg}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Relatório salvo em '{output_path}'")
