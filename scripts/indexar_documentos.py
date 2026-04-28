"""
Pipeline de indexação dos documentos institucionais da UFPA.

Lê os PDFs de documents/, gera embeddings com HuggingFace e persiste
o índice em storage/. Não usa Gemini — apenas embeddings locais.

Uso:
    python scripts/indexar_documentos.py           # indexa documentos novos
    python scripts/indexar_documentos.py --reset   # apaga e reindexa tudo
"""
import argparse
import os
import sys

# Garante que imports relativos ao root do projeto funcionem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import EMBEDDING_MODEL, STORAGE_DIR, CHROMA_COLLECTION_NAME
from utils.logger import logger

DOCUMENTS_DIR = "./documents"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Indexa documentos no ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga a coleção existente e reindexa todos os documentos.",
    )
    return parser.parse_args()


def setup_embeddings():
    logger.info(f"⚙️  Carregando embeddings: {EMBEDDING_MODEL}")
    Settings.llm = None  # indexação não precisa de LLM
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    logger.info("✅ Embeddings carregados.")


def load_documents():
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        logger.error(f"❌ Diretório '{DOCUMENTS_DIR}' vazio ou inexistente.")
        sys.exit(1)

    logger.info(f"📄 Lendo documentos de '{DOCUMENTS_DIR}'...")
    documents = SimpleDirectoryReader(
        DOCUMENTS_DIR,
        recursive=True,
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    logger.info(f"✅ {len(documents)} documentos carregados.")
    return documents


def setup_storage(reset: bool):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    logger.info(f"📦 Conectando ao ChromaDB em '{STORAGE_DIR}'...")

    chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)

    if reset:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("🧹 Coleção antiga removida (--reset).")
        except Exception:
            logger.info("ℹ️  Nenhuma coleção anterior encontrada.")

    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    docstore = SimpleDocumentStore()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )
    return storage_context, chroma_collection


def run_pipeline(documents, storage_context):
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    logger.info("🔪 Executando pipeline de ingestão...")
    pipeline = IngestionPipeline(
        transformations=[node_parser, Settings.embed_model],
        vector_store=storage_context.vector_store,
        docstore=storage_context.docstore,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    logger.info(f"✅ {len(nodes)} chunks gerados.")
    return nodes


def build_index(nodes, storage_context):
    logger.info("🧠 Construindo índice vetorial...")
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    logger.info("✅ Índice construído.")


def persist(storage_context):
    logger.info(f"💾 Persistindo em '{STORAGE_DIR}'...")
    storage_context.persist(persist_dir=STORAGE_DIR)
    logger.info("✅ Persistência concluída.")


def print_summary(documents, nodes, chroma_collection):
    print("\n" + "=" * 60)
    print("📊 RESUMO DA INDEXAÇÃO")
    print("=" * 60)
    print(f"  Documentos lidos:        {len(documents)}")
    print(f"  Chunks gerados:          {len(nodes)}")
    print(f"  Vetores no ChromaDB:     {chroma_collection.count()}")
    print(f"  Storage:                 {STORAGE_DIR}/")
    print("=" * 60)
    print("\n✅ Indexação finalizada. Atualize a imagem Docker para refletir as mudanças.\n")


def main():
    args = parse_args()

    setup_embeddings()
    documents = load_documents()
    storage_context, chroma_collection = setup_storage(reset=args.reset)
    nodes = run_pipeline(documents, storage_context)
    build_index(nodes, storage_context)
    persist(storage_context)
    print_summary(documents, nodes, chroma_collection)


if __name__ == "__main__":
    main()
