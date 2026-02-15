"""
Implementação do retriever híbrido (Vetorial + BM25).
"""
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from utils.logger import logger


class HybridRetriever(BaseRetriever):
    """
    Combina busca vetorial (semântica) e BM25 (palavras-chave).
    
    A busca híbrida melhora a precisão ao unir:
    - Busca vetorial: entende significado e contexto
    - BM25: encontra correspondências exatas de termos
    """
    
    def __init__(self, vector_retriever, bm25_retriever):
        """
        Inicializa o retriever híbrido.
        
        Args:
            vector_retriever: Retriever de busca vetorial
            bm25_retriever: Retriever BM25
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        """
        Executa busca híbrida e retorna nós únicos.
        
        Args:
            query_bundle: Query do usuário
            
        Returns:
            Lista de nós recuperados (sem duplicatas)
        """
        query_text = query_bundle.query_str
        logger.info(f"🔍 Buscando: '{query_text[:50]}...'")
        
        # Busca vetorial (semântica)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        logger.info(f"  ✓ Vector: {len(vector_nodes)} nodes")
        
        # Busca BM25 (palavras-chave)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        logger.info(f"  ✓ BM25: {len(bm25_nodes)} nodes")

        # União eliminando duplicatas por node_id
        all_nodes = {n.node.node_id: n for n in vector_nodes + bm25_nodes}
        logger.info(f"  ✓ Total único: {len(all_nodes)} nodes")
        
        return list(all_nodes.values())