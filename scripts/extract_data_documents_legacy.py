import os
import logging
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# ---------------- LOG ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- MODELS ----------------
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_data_extraction():
    input_dir = "./documents"
    storage_dir = "./storage"
    collection_name = "institucional_db"

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"❌ Diretório {input_dir} vazio ou inexistente.")
        return

    print("📄 Lendo documentos...")
    documents = SimpleDirectoryReader(input_dir).load_data()
    print(f"✓ Documentos carregados: {len(documents)}")
    
    total_chars = sum(len(doc.text) for doc in documents)
    print(f"✓ Total de caracteres nos documentos: {total_chars:,}")

    # ---------------- CRIAR DIRETÓRIO DE STORAGE ----------------
    os.makedirs(storage_dir, exist_ok=True)
    print(f"📁 Diretório de storage: {storage_dir}")

    # ---------------- CHROMA ----------------
    print("📦 Conectando ao ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=storage_dir)

    try:
        chroma_client.delete_collection(collection_name)
        print("🧹 Coleção antiga removida.")
    except:
        print("ℹ Primeira execução, nenhuma coleção anterior.")

    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # ---------------- STORAGE CONTEXT (SEM CARREGAR EXISTENTE) ----------------
    print("🗂️ Criando novo storage context...")
    docstore = SimpleDocumentStore()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )

    # ---------------- NODE PARSER ----------------
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=128
    )
    
    # ---------------- PIPELINE DE INGESTÃO ----------------
    print("🔪 Processando documentos com pipeline...")
    
    pipeline = IngestionPipeline(
        transformations=[node_parser, Settings.embed_model],
        vector_store=vector_store,
        docstore=docstore,
    )
    
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"✓ Nodes processados pelo pipeline: {len(nodes)}")

    # ---------------- ADICIONAR NODES AO DOCSTORE ANTES DO ÍNDICE ----------------
    print("\n🔍 Adicionando nodes processados ao docstore...")
    for node in nodes:
        storage_context.docstore.add_documents([node])
    
    nodes_in_docstore = len(storage_context.docstore.docs)
    print(f"  ✓ {nodes_in_docstore} nodes adicionados ao docstore")
    
    # ---------------- CRIAR ÍNDICE ----------------
    print("\n🧠 Criando índice...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # ---------------- PERSISTÊNCIA ----------------
    print("\n💾 Persistindo dados...")
    storage_context.persist(persist_dir=storage_dir)
    print("✓ Persistência concluída")

    # ---------------- VALIDAÇÃO PÓS-PERSISTÊNCIA ----------------
    print("\n🔍 Validando persistência...")
    
    # Recarregar do disco para validar
    try:
        loaded_storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=storage_dir
        )
        loaded_nodes = loaded_storage_context.docstore.docs
        print(f"✓ Nodes carregados do disco: {len(loaded_nodes)}")
    except Exception as e:
        print(f"❌ Erro ao recarregar: {e}")
        loaded_nodes = {}

    # Estatísticas
    nodes_with_text_count = 0
    total_node_chars = 0
    for node_id, node in loaded_nodes.items():
        if hasattr(node, 'text') and node.text:
            nodes_with_text_count += 1
            total_node_chars += len(node.text)
    
    print(f"📊 Nodes com texto: {nodes_with_text_count}")
    print(f"📊 Total de caracteres nos nodes: {total_node_chars:,}")

    print("\n" + "=" * 60)
    print("✅ INDEXAÇÃO FINALIZADA")
    print("=" * 60)
    print(f"Documentos originais: {len(documents)}")
    print(f"Nodes processados: {len(nodes)}")
    print(f"Nodes no docstore (memória): {nodes_in_docstore}")
    print(f"Nodes carregados do disco: {len(loaded_nodes)}")
    print(f"Nodes com texto: {nodes_with_text_count}")
    print(f"Documentos no ChromaDB: {chroma_collection.count()}")
    print(f"Storage: {storage_dir}")

    # Verificar arquivos criados
    print(f"\n📁 Arquivos criados em {storage_dir}:")
    json_files = []
    for file in sorted(os.listdir(storage_dir)):
        file_path = os.path.join(storage_dir, file)
        if os.path.isfile(file_path):
            if file.endswith('.json'):
                json_files.append(file)
                size = os.path.getsize(file_path)
                print(f"  ✓ {file} ({size:,} bytes)")
    
    # Análise detalhada do docstore.json
    import json
    docstore_path = os.path.join(storage_dir, "docstore.json")
    if os.path.exists(docstore_path):
        with open(docstore_path, 'r', encoding='utf-8') as f:
            docstore_data = json.load(f)
            
        print(f"\n📝 Análise do docstore.json:")
        
        # Verificar todas as chaves
        for key in docstore_data.keys():
            if isinstance(docstore_data[key], dict):
                item_count = len(docstore_data[key])
                print(f"  - {key}: {item_count} items")
                
                # Se for docstore/data, mostrar preview
                if key == "docstore/data" and item_count > 0:
                    first_node_id = list(docstore_data[key].keys())[0]
                    first_node = docstore_data[key][first_node_id]
                    
                    # Verificar estrutura do node
                    node_keys = list(first_node.keys()) if isinstance(first_node, dict) else []
                    print(f"    Chaves do node: {node_keys}")
                    
                    if isinstance(first_node, dict) and 'text' in first_node:
                        text_preview = first_node['text'][:100]
                        print(f"    Preview: {text_preview}...")
                    elif isinstance(first_node, dict) and '__data__' in first_node:
                        print(f"    Node serializado encontrado")
                        if 'text' in first_node['__data__']:
                            text_preview = first_node['__data__']['text'][:100]
                            print(f"    Preview: {text_preview}...")
        
        # Status final
        docstore_nodes = docstore_data.get('docstore/data', {})
        if len(docstore_nodes) > 0:
            print(f"\n✅ SUCCESS! Docstore contém {len(docstore_nodes)} nodes")
        else:
            print(f"\n⚠️ AVISO: docstore/data está vazio")
            print(f"   Total de chaves no JSON: {len(docstore_data)}")
    else:
        print(f"\n❌ ERRO: docstore.json não foi criado!")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_data_extraction()