"""
Script de diagnóstico para entender o que está sendo indexado e recuperado.
"""
import os
import json
import chromadb
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configurar modelos
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

STORAGE_DIR = "./storage"
COLLECTION_NAME = "institucional_db"

def analyze_docstore():
    """Analisa o conteúdo do docstore."""
    print("=" * 80)
    print("📊 ANÁLISE DO DOCSTORE")
    print("=" * 80)
    
    docstore_path = os.path.join(STORAGE_DIR, "docstore.json")
    
    if not os.path.exists(docstore_path):
        print("❌ docstore.json não encontrado!")
        return
    
    with open(docstore_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes_data = data.get('docstore/data', {})
    print(f"\n📝 Total de nodes no docstore: {len(nodes_data)}")
    
    # Analisar primeiros 5 nodes
    print("\n🔍 AMOSTRA DOS PRIMEIROS 5 NODES:")
    print("-" * 80)
    
    for i, (node_id, node_data) in enumerate(list(nodes_data.items())[:5]):
        if '__data__' in node_data:
            node_content = node_data['__data__']
            text = node_content.get('text', '')
            metadata = node_content.get('metadata', {})
            
            print(f"\n[Node {i+1}] ID: {node_id[:20]}...")
            print(f"Tipo: {node_data.get('__type__', 'N/A')}")
            print(f"Metadados: {metadata}")
            print(f"Tamanho do texto: {len(text)} caracteres")
            print(f"Preview do texto:\n{text[:300]}...")
            print("-" * 80)

def test_search_query(query: str):
    """Testa uma busca específica."""
    print("\n" + "=" * 80)
    print(f"🔍 TESTANDO BUSCA: '{query}'")
    print("=" * 80)
    
    # Conectar ao ChromaDB
    chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    
    print(f"\n📦 ChromaDB - Total de documentos: {chroma_collection.count()}")
    
    # Carregar índice
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=STORAGE_DIR
    )
    
    index = load_index_from_storage(storage_context)
    
    # Criar retriever simples
    from llama_index.core.retrievers import VectorIndexRetriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )
    
    # Buscar
    print(f"\n🔎 Executando busca vetorial...")
    nodes = retriever.retrieve(query)
    
    print(f"\n✓ Recuperados {len(nodes)} nodes:")
    print("-" * 80)
    
    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') else 'N/A'
        text = node.node.text if hasattr(node.node, 'text') else str(node.node)
        metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
        
        print(f"\n[Resultado {i}] Score: {score}")
        print(f"Metadados: {metadata}")
        print(f"Texto ({len(text)} chars):\n{text[:400]}...")
        print("-" * 80)

def search_raw_text_in_nodes(search_term: str):
    """Busca texto bruto nos nodes (sem embeddings)."""
    print("\n" + "=" * 80)
    print(f"🔎 BUSCA TEXTUAL BRUTA: '{search_term}'")
    print("=" * 80)
    
    docstore_path = os.path.join(STORAGE_DIR, "docstore.json")
    
    with open(docstore_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes_data = data.get('docstore/data', {})
    matches = []
    
    for node_id, node_data in nodes_data.items():
        if '__data__' in node_data:
            text = node_data['__data__'].get('text', '').lower()
            if search_term.lower() in text:
                matches.append({
                    'id': node_id,
                    'text': node_data['__data__'].get('text', ''),
                    'metadata': node_data['__data__'].get('metadata', {})
                })
    
    print(f"\n✓ Encontrados {len(matches)} nodes com '{search_term}'")
    
    if matches:
        print("\n🎯 PRIMEIROS 3 MATCHES:")
        for i, match in enumerate(matches[:3], 1):
            print(f"\n[Match {i}]")
            print(f"Metadados: {match['metadata']}")
            print(f"Texto:\n{match['text'][:500]}...")
            print("-" * 80)
    else:
        print(f"\n⚠️ NENHUM node contém o termo '{search_term}'!")
        print("Isso indica que o texto NÃO foi indexado corretamente.")

def main():
    """Executa diagnóstico completo."""
    print("\n🚀 INICIANDO DIAGNÓSTICO COMPLETO\n")
    
    # 1. Analisar docstore
    analyze_docstore()
    
    # 2. Busca textual por termos esperados
    print("\n" + "=" * 80)
    print("📋 VERIFICANDO PRESENÇA DE TERMOS-CHAVE")
    print("=" * 80)
    
    search_terms = [
        "primeiro semestre",
        "bloco I",
        "disciplina",
        "engenharia da computação",
        "cálculo",
        "programação",
        "álgebra"
    ]
    
    for term in search_terms:
        search_raw_text_in_nodes(term)
    
    # 3. Testar busca vetorial
    test_queries = [
        "quais disciplinas do primeiro semestre",
        "disciplinas bloco I",
        "matérias do primeiro período"
    ]
    
    for query in test_queries:
        test_search_query(query)
    
    print("\n" + "=" * 80)
    print("✅ DIAGNÓSTICO CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    main()