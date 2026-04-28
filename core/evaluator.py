"""
Módulo de avaliação quantitativa utilizando o framework RAGAS.

CHANGELOG:
- LLM Juiz migrado para ChatGoogleGenerativeAI (google-genai, API v1 estável).
- Corrigido formato do nome do modelo: "gemini-2.0-flash" (sem prefixo "models/").
- API key centralizada via config.py (nunca hardcodada no código).
- Imports das métricas RAGAS atualizados para a sintaxe de classes instanciadas.
"""
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

# Métricas instanciadas como classes (sintaxe correta pós-ragas 0.1.x)
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from core.engine import InstitutionalHybridBot
from utils.logger import logger
from config import GOOGLE_API_KEY, JUDGE_MODEL


class RAGEvaluator:
    """Classe para validação estatística do motor RAG com framework RAGAS."""

    def __init__(self):
        logger.info("🧪 Inicializando Avaliador RAGAS...")

        # Bot de produção (usa gemini-2.5-flash via LangChainLLM + LlamaIndex)
        self.bot = InstitutionalHybridBot()

        # -----------------------------------------------------------------------
        # LLM Juiz: gemini-2.0-flash (~1.500 req/dia no plano gratuito)
        # NOTA: ChatGoogleGenerativeAI aceita o nome curto do modelo SEM "models/"
        #       Ex.: "gemini-2.0-flash"  ✅
        #            "models/gemini-2.0-flash"  ❌ (causa 404 no langchain-google-genai)
        # -----------------------------------------------------------------------
        self.judge_llm = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,   # determinismo total para o juiz
            max_retries=3,
        )

        # Embeddings para a métrica AnswerRelevancy (cosseno entre pergunta e resposta)
        # self.judge_embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/text-embedding-004",
        #     google_api_key=GOOGLE_API_KEY,
        # )
        
        logger.info("⚙️ Carregando Embeddings locais para o Juiz...")
        self.judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        self.metrics = [
            Faithfulness(),       # Resposta é fiel ao contexto recuperado?
            AnswerRelevancy(),    # Resposta é relevante para a pergunta?
            ContextPrecision(),   # Contexto recuperado contém a ground-truth?
        ]

    def run_evaluation(self, test_set: list) -> pd.DataFrame:
        """
        Executa as perguntas de teste e gera as métricas.
        """
        user_inputs = []
        responses = []
        retrieved_contexts = []
        references = []

        logger.info(f"▶️ Iniciando teste com {len(test_set)} questões...")

        for item in test_set:
            q = item["user_input"]
            logger.info(f"Perguntando: {q}")
            
            try:
                response = self.bot.query_engine.query(q)
                ans = str(response)
                ctx = [n.node.get_content() for n in response.source_nodes]
                
            except Exception as e:
                logger.error(f"Erro ao processar '{q}': {e}")
                ans = "Erro na geração"
                ctx = ["Erro"]

            user_inputs.append(q)
            responses.append(ans)
            retrieved_contexts.append(ctx)
            references.append(item["reference"])

        data = {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references
        }
        dataset = Dataset.from_dict(data)

        logger.info("🧠 Calculando métricas RAGAS (Isto pode demorar devido aos limites da API gratuita)...")
        
        # 2. CONFIGURAÇÃO DE LENTIDÃO PARA NÃO ESGOTAR A API
        api_config = RunConfig(
            max_workers=1,       # Força o RAGAS a avaliar uma coisa de cada vez
            max_retries=15,      # Tenta várias vezes se o Google bloquear
            max_wait=60          # Espera até 60 segundos entre tentativas
        )
        
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            embeddings=self.judge_embeddings,
            run_config=api_config # <-- APLICANDO A CONFIGURAÇÃO AQUI
        )
        
        return result.to_pandas()

# ---------------------------------------------------------------------------
# Conjunto de testes para o TCC — cobrindo as três categorias de documentos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # ==========================================
        # NÍVEL 1: PERGUNTAS DIRETAS (Factoides)
        # Testam a capacidade do bot de encontrar números, horas e regras simples.
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
        # NÍVEL 2: PERGUNTAS PROCEDIMENTAIS
        # Testam a capacidade de extrair passos e instruções sem omitir etapas.
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
        # NÍVEL 3: RACIOCÍNIO LÓGICO E CONDICIONAIS
        # Testam a capacidade do bot não ser enganado por falsas premissas.
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
        }
    ]

    evaluator = RAGEvaluator()
    df = evaluator.run_evaluation(test_cases)

    print("\n" + "=" * 80)
    print("📊 RESULTADOS DA AVALIAÇÃO CIENTÍFICA (RAGAS)")
    print("=" * 80)

    cols_to_show = ["user_input", "faithfulness", "answer_relevancy", "context_precision"]
    print(df[cols_to_show].to_string(index=False))

    output_path = "resultados_tcc.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Relatório completo salvo em '{output_path}'")