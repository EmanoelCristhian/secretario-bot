"""
Motor de busca híbrida genérico para documentos institucionais.
"""
import os
import chromadb

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.gemini import Gemini


from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_TOP_K,
    SIMILARITY_CUTOFF,
    STORAGE_DIR,
    CHROMA_COLLECTION_NAME
)
from core.retriever import HybridRetriever
from core.prompts import PromptTemplates, ResponseValidator
from utils.logger import logger


class InstitutionalHybridBot:
    """
    Motor de busca híbrida para documentos institucionais da UFPA.
    """
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.prompt_templates = PromptTemplates()
        self.response_validator = ResponseValidator()
        self._configure_llm()
        self.query_engine = self._setup_hybrid_engine()

    def _configure_llm(self):
        """Configura o modelo de linguagem (Gemini) e embeddings."""
        logger.info(f"⚙️ Configurando LLM: {LLM_MODEL}")
        
        # Criar instância do LLM Gemini
        self.llm = Gemini(
            model=LLM_MODEL,
            api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            system_prompt=self.prompt_templates.build_system_message()
        )
        
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        logger.info(f"✅ LLM configurado: {LLM_MODEL}")

    def _create_custom_prompt_template(self):
        """
        Prompt otimizado para extração exaustiva. 
        Remove a exigência de nomes em inglês e créditos para evitar que o LLM ignore dados.
        """
        template_str = """Você é um assistente acadêmico da UFPA.
Sua tarefa é extrair e listar informações exclusivamente dos documentos fornecidos.

### REGRAS OBRIGATÓRIAS:
1. **FOCO NO POSITIVO**: Se a informação for encontrada em qualquer parte do contexto, ignore os trechos que não a mencionam. NÃO diga "não encontrei" se a informação aparecer em pelo menos um lugar.
2. **SEM COMENTÁRIOS EXTRAS**: Não adicione informações sobre o que NÃO está no documento (como menções a outros semestres ou atividades de extensão) se o utilizador não perguntou por isso.
3. **EXAUSTIVIDADE**: Liste TODAS as disciplinas e cargas horárias encontradas para o bloco solicitado.
4. **FIDELIDADE**: Transcreva exatamente como aparece (Ex: "Física 60" vira "Física - 60 horas").
5. **FONTE**: Cite o ficheiro de origem no final.

### CONTEXTO DOS DOCUMENTOS:
{context_str}

### PERGUNTA DO USUÁRIO:
{query_str}

### RESPOSTA OBJETIVA (Baseada apenas nos dados encontrados):"""

        return PromptTemplate(template_str)

    def _setup_hybrid_engine(self):
        """Configura o motor de busca híbrida."""
        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError(f"Diretório '{self.storage_dir}' não encontrado.")

        logger.info("📦 Conectando ao ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=self.storage_dir)
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self.storage_dir
        )

        index = load_index_from_storage(storage_context)
        nodes = self._get_valid_nodes(index, chroma_collection)
        
        # Aumentamos o top_k para garantir que listas longas de disciplinas sejam capturadas
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=SIMILARITY_TOP_K)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=SIMILARITY_TOP_K)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        text_qa_template = self._create_custom_prompt_template()
        
        # TREE_SUMMARIZE é ideal para consolidar listas espalhadas em vários documentos
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            text_qa_template=text_qa_template,
            verbose=True
        )

        logger.info("✅ Engine híbrida pronta!")

        return RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.3) # Cutoff reduzido para capturar toda a lista
            ]
        )

    def _get_valid_nodes(self, index, chroma_collection):
        """Recupera nós para o BM25."""
        nodes = list(index.docstore.docs.values())
        if not nodes:
            all_ids = chroma_collection.get(limit=2000)['ids']
            vector_retriever_temp = VectorIndexRetriever(index=index, similarity_top_k=min(len(all_ids), 200))
            nodes = [n.node for n in vector_retriever_temp.retrieve("recuperar")]

        return [n for n in nodes if hasattr(n, 'text') and n.text and n.text.strip()]

    def query(self, text: str) -> str:
        """Processa a consulta com validação."""
        logger.info(f"💬 Query recebida: '{text[:100]}...'")
        try:
            # 1. Execução da busca e síntese usando o texto original
            response = self.query_engine.query(text)
            response_text = str(response)
            
            # 2. Validação de segurança
            validated_response = self.response_validator.validate_response(response_text, text)
            
            if self.response_validator.detect_hallucination_indicators(validated_response):
                logger.warning("⚠️ Possível alucinação detectada")
            
            return validated_response
        except Exception as e:
            logger.error(f"❌ Erro no motor de busca: {e}", exc_info=True)
            raise

    def get_context_for_query(self, text: str, top_k: int = 15) -> str:
        """Recupera apenas o contexto para análise de debug."""
        try:
            # Recupera utilizando o texto original
            nodes = self.query_engine.retriever.retrieve(text)
            context_parts = [f"📝 Query original: {text}\n{'='*40}\n"]
            
            for i, node in enumerate(nodes[:top_k], 1):
                source = node.node.metadata.get('file_name', 'N/A') if hasattr(node.node, 'metadata') else 'N/A'
                context_parts.append(f"[Node {i}] Fonte: {source}\nConteúdo: {node.node.text[:400]}...\n{'-'*40}")
            
            return "\n".join(context_parts)
        except Exception as e:
            return f"Erro ao recuperar contexto: {e}"