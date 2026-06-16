# Secretário Bot FCT/UFPA

Assistente virtual para o Telegram que responde dúvidas acadêmicas dos alunos da Faculdade de Engenharia da Computação e Telecomunicações (FCT) da Universidade Federal do Pará (UFPA), utilizando RAG (*Retrieval-Augmented Generation*) sobre um corpus de documentos institucionais em PDF.

Desenvolvido como Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação — FCT/UFPA.

---

## Demonstração

O bot responde perguntas como:

- *"Quais disciplinas fazem parte do 3º bloco de Engenharia da Computação?"*
- *"Qual é o prazo para entrega do relatório de estágio?"*
- *"Quantos créditos preciso para integralizar atividades complementares?"*
- *"Como funciona a matrícula em TCC I?"*

---

## Arquitetura

```
Usuário (Telegram)
       │
       ▼
 GreetingDetector          ← classifica: saudação / pergunta / small talk
       │
       ▼
ConversationHistory        ← prepend dos últimos 5 turnos à query
       │
       ▼
 HybridRetriever
  ├── VectorIndexRetriever  ← ChromaDB, embeddings BAAI/bge-small-en-v1.5, distância L2
  ├── BM25Retriever         ← BM25+ Lucene sobre docstore em memória
  └── RRF (k=60)           ← Reciprocal Rank Fusion combina os dois rankings
       │
       ▼
TREE_SUMMARIZE             ← Gemini 2.5 Flash (T=0.1) sintetiza os chunks
       │
       ▼
ResponseValidator          ← valida, remove prefixos, intercepta inglês do LlamaIndex
       │
       ▼
 Resposta ao usuário
```

### Estrutura de arquivos

```
secretario-bot/
├── main.py                          # Entrada: inicializa bot e engine
├── bot/
│   ├── handlers.py                  # Handlers Telegram (aiogram)
│   └── messages.py                  # Templates de mensagens
├── core/
│   ├── engine.py                    # InstitutionalHybridBot (orquestra tudo)
│   ├── retriever.py                 # HybridRetriever com RRF
│   ├── prompts.py                   # Prompt de síntese + ResponseValidator
│   └── evaluator.py                 # Avaliação RAGAS (hybrid/vector/bm25)
├── utils/
│   ├── greeting_detector.py         # Classificação de intenção por regex
│   ├── conversation_history.py      # Buffer circular de histórico (deque)
│   └── logger.py
├── config/
│   └── settings.py                  # Constantes e variáveis de ambiente
├── scripts/
│   └── indexar_documentos.py        # Pipeline de indexação offline
├── docs/
│   ├── METODOLOGIA.md               # Metodologia acadêmica completa
│   └── formulario_avaliacao_bot.docx
├── Dockerfile
├── docker-compose.yml
└── railway.json
```

---

## Metodologia

### Fase 1 — Indexação (offline)

```bash
python scripts/indexar_documentos.py        # indexa documentos novos
python scripts/indexar_documentos.py --reset  # reindexa tudo do zero
```

1. **Leitura** dos PDFs com `SimpleDirectoryReader` (extrai texto + metadados de origem)
2. **Chunking** com `SentenceSplitter` — `chunk_size=1024`, `chunk_overlap=128`
3. **Embeddings** com `BAAI/bge-small-en-v1.5` (384 dimensões, local, sem API)
4. **Persistência** no ChromaDB (SQLite3, coleção `institucional_db`)

Resultado: 898 nós no docstore, 455 vetores no ChromaDB.

### Fase 2 — Recuperação e síntese (online)

Para cada query do usuário:

1. **Busca vetorial**: top-20 chunks por distância L2 no espaço de 384d
2. **Busca lexical**: top-20 chunks por BM25+ Lucene (k₁=1.5, b=0.75, δ=0.5)
3. **RRF**: combina os dois rankings pela fórmula `score(d) = Σ 1/(60 + rank)`
4. **Síntese**: os chunks fundidos são enviados ao Gemini 2.5 Flash via `TREE_SUMMARIZE`

### Corpus

31 documentos PDF institucionais da FCT/UFPA:

| Categoria | Documentos |
|---|---|
| Ementas — Engenharia da Computação | 8 PDFs (blocos I–VIII) |
| Ementas — Engenharia de Telecomunicações | 10 PDFs (blocos I–X) |
| Regulamentos FCT (TCC, Estágio, Ativ. Complementares) | 5 PDFs |
| Documentos institucionais | 8 PDFs |

---

## Resultados

Avaliação com o framework **RAGAS** usando Gemini 2.5 Flash como LLM juiz (T=0.0).

### Baseline (pré-melhorias)

Coletado antes da implementação do RRF e do histórico de conversa, com 15 questões:

| Métrica | Resultado |
|---|---|
| Faithfulness | **0.896** |
| Answer Relevancy | **0.888** |
| Context Precision | **0.280** |

O **Context Precision baixo (0.280)** evidenciou que os documentos relevantes eram recuperados, mas não posicionados nas primeiras posições do ranking — motivando a implementação do RRF.

### Avaliação comparativa (após melhorias)

Executar com: `python -m core.evaluator`

| Configuração | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| `hybrid` (RRF) | — | — | — | — |
| `vector` | — | — | — | — |
| `bm25` | — | — | — | — |

> Tabela a ser preenchida após execução da avaliação comparativa completa.

---

## Como rodar

### Pré-requisitos

- Python 3.12+
- Chave de API Google Gemini (`GOOGLE_API_KEY`)
- Token do bot Telegram (`TELEGRAM_TOKEN`)

### 1. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Editar .env com TELEGRAM_TOKEN e GOOGLE_API_KEY
```

Conteúdo do `.env`:
```
TELEGRAM_TOKEN=seu_token_aqui
GOOGLE_API_KEY=sua_chave_aqui
```

### 2. Instalar dependências

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Indexar documentos

```bash
# Coloque os PDFs em documents/
python scripts/indexar_documentos.py
```

### 4. Rodar o bot

```bash
python main.py
```

### Docker (simula produção)

```bash
docker compose up --build
```

---

## Deploy (Railway)

O projeto está configurado para deploy automático via `railway.json`. A imagem usa `python:3.13-slim` com PyTorch CPU-only para reduzir o tamanho do build.

Variáveis obrigatórias no Railway:
- `TELEGRAM_TOKEN`
- `GOOGLE_API_KEY`

---

## Avaliação

```bash
# Avaliação comparativa (hybrid vs vector vs bm25) — requer GOOGLE_API_KEY válida
python -m core.evaluator

# Avaliar apenas um modo
python -m core.evaluator hybrid
python -m core.evaluator vector
python -m core.evaluator bm25
```

Resultados salvos em `resultados_tcc_comparativo.csv`.

---

## Stack tecnológica

| Componente | Tecnologia |
|---|---|
| Interface | Telegram Bot API (aiogram 3.27.0) |
| Orquestração RAG | LlamaIndex 0.14.21 |
| LLM de geração | Google Gemini 2.5 Flash |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Banco vetorial | ChromaDB 1.5.8 (SQLite3) |
| Busca lexical | BM25+ via bm25s 0.3.6 |
| Avaliação | RAGAS 0.4.3 |
| Deploy | Docker + Railway |

---

## Documentação acadêmica

- [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md) — metodologia completa com fórmulas LaTeX, fluxograma Mermaid e limitações
- [`resultados_tcc_baseline.csv`](resultados_tcc_baseline.csv) — resultados baseline pré-RRF
- [`docs/formulario_avaliacao_bot.docx`](docs/formulario_avaliacao_bot.docx) — formulário de avaliação humana com mapeamento RAGAS

---

## Autor

**Emanoel Marinho** — Engenharia da Computação, FCT/UFPA
