"""
Implementação do retriever híbrido (Vetorial + BM25) com Reciprocal Rank Fusion.
"""
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from utils.logger import logger


class HybridRetriever(BaseRetriever):
    """
    Combina busca vetorial (semântica) e BM25 (palavras-chave) via RRF.

    Reciprocal Rank Fusion pondera cada nó pela sua posição nos dois rankings
    independentes (score = 1/(k + rank)), tornando o resultado mais robusto do
    que a simples união — nós bem posicionados em ambas as buscas sobem no
    ranking final sem depender da escala absoluta dos scores originais.
    """

    RRF_K = 60  # constante de suavização padrão da literatura RRF

    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        query_text = query_bundle.query_str
        logger.info(f"🔍 Buscando: '{query_text[:50]}...'")

        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        logger.info(f"  ✓ Vector: {len(vector_nodes)} nodes")

        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        logger.info(f"  ✓ BM25: {len(bm25_nodes)} nodes")

        fused = self._reciprocal_rank_fusion(vector_nodes, bm25_nodes)
        logger.info(f"  ✓ RRF total: {len(fused)} nodes")

        return fused

    def _reciprocal_rank_fusion(
        self,
        vector_nodes: list[NodeWithScore],
        bm25_nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """Combina dois rankings via RRF e devolve nós ordenados por score."""
        k = self.RRF_K
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for rank, nws in enumerate(vector_nodes):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            node_map[nid] = nws

        for rank, nws in enumerate(bm25_nodes):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            node_map[nid] = nws

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        result = []
        for nid in sorted_ids:
            nws = node_map[nid]
            result.append(NodeWithScore(node=nws.node, score=rrf_scores[nid]))

        return result