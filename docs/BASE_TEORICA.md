# Base Teórica

Referência estruturada dos conceitos que fundamentam o sistema, organizada na mesma ordem da Revisão Bibliográfica do TCC. Cada seção apresenta o conceito, sua relação direta com o projeto e as referências utilizadas.

---

## 1. Processamento de Linguagem Natural (PLN)

O Processamento de Linguagem Natural é um campo interdisciplinar que combina linguística computacional e aprendizado de máquina para permitir que sistemas computacionais processem e gerem linguagem humana (CHEN et al., 2024).

### Evolução dos paradigmas

| Paradigma | Abordagem | Limitação |
|---|---|---|
| Simbólico/regras | Gramáticas formais, análise sintática top-down (CKY) | Dependência de regras rígidas; falha em linguagem informal |
| Estatístico/ML | Naive Bayes, SVM, n-gramas | Representações esparsas; sem semântica |
| Redes neurais (PLMs) | BERT, word2vec; representações context-aware | Fine-tuning caro; conhecimento estático |
| LLMs | GPT, PaLM, Gemini; capacidades emergentes | Alucinações; knowledge cutoff; custo computacional |

**Relevância para o projeto:** a transição do paradigma sintático para o semântico justifica a adoção de embeddings densos (busca vetorial) em substituição a buscas por palavras-chave, e a escolha de um LLM (Gemini 2.5 Flash) para síntese das respostas.

**Referências:** CHEN et al. (2024); JURAFSKY; MARTIN (2019).

---

## 2. Modelos de Linguagem de Grande Escala (LLMs)

### 2.1 Gerações de modelos de linguagem

Zhao et al. (2024) identificam quatro gerações de modelos de linguagem:

1. **Estatísticos** — cálculo probabilístico para tarefas específicas
2. **Neurais** — aprendizado de representações agnósticas à tarefa
3. **PLMs** (*Pre-trained Language Models*) — representações context-aware com fine-tuning (BERT, GPT-2)
4. **LLMs** — escala massiva + capacidades emergentes (GPT-3, PaLM, Gemini)

### 2.2 Capacidades emergentes

O aumento de escala (*scaling*) obedece a leis de escala (KAPLAN et al., 2020): modelos com bilhões de parâmetros exibem comportamentos qualitativamente distintos de versões menores, como o aprendizado no contexto (*in-context learning*) e resolução de tarefas *few-shot* (WEI et al., 2022a).

### 2.3 Paradigmas operacionais (Qin et al., 2024)

| Paradigma | Descrição | Uso no projeto |
|---|---|---|
| **Parameter-Frozen** | Modelo usado sem modificar pesos; apenas prompts | ✅ Este projeto — zero fine-tuning |
| **Parameter-Tuning** | Fine-tuning completo ou eficiente (LoRA) | ✗ Não utilizado |

**Justificativa da escolha:** o paradigma de parâmetros congelados via API elimina o custo de infraestrutura de treinamento e é adequado para domínios com documentação estruturada, onde o RAG supre a necessidade de conhecimento específico.

### 2.4 Limitações dos LLMs puros

- **Alucinações:** geração de informações factualmente falsas (JI et al., 2023) — crítico em contexto acadêmico onde prazos e normas são verificáveis
- **Knowledge cutoff:** conhecimento limitado à data de treinamento
- **Custo computacional:** APIs pagas por token consumido

**Referências:** ZHAO et al. (2024); QIN et al. (2024); BROWN et al. (2020); YANG et al. (2024); WEI et al. (2022a); JI et al. (2023); KAPLAN et al. (2020).

---

## 3. Retrieval-Augmented Generation (RAG)

### 3.1 Definição e origem

Proposto por Lewis et al. (2020), o RAG combina um modelo de linguagem com um sistema de recuperação de informação externo. O modelo não depende exclusivamente do conhecimento de treinamento — antes de gerar uma resposta, recupera trechos relevantes de uma base documental.

**Fluxo básico:**
```
Query → Retriever → [chunks relevantes] → LLM → Resposta fundamentada
```

**Vantagens sobre LLMs puros:**
- Respostas ancoradas em evidências documentais (↓ alucinações)
- Base de conhecimento atualizável sem re-treinamento
- Auditabilidade das fontes

### 3.2 Evolução arquitetural (Gao et al., 2024)

| Tipo | Característica | Limitação superada |
|---|---|---|
| **Naive RAG** | Recuperação linear → síntese | Chunks irrelevantes degradam resposta |
| **Advanced RAG** | Pré/pós-recuperação otimizados | Melhora relevância dos chunks |
| **Modular RAG** | Módulos independentes de busca e validação | Máxima flexibilidade |

**Este projeto:** implementa **Advanced RAG** com pré-recuperação por dois retrievers independentes (vetorial + BM25) e pós-recuperação por RRF.

### 3.3 Posicionamento em relação a outras técnicas

| Técnica | Conhecimento externo | Adaptação do modelo | Quando usar |
|---|---|---|---|
| Prompt Engineering | Baixo | Nenhuma | Tarefas genéricas |
| RAG | Alto (dinâmico) | Nenhuma | Domínios documentais específicos ✅ |
| Fine-tuning | Baixo (estático) | Alta | Estilo/comportamento específico |
| Híbrido (RAG + FT) | Alto | Alta | Sistemas corporativos complexos |

**Referências:** LEWIS et al. (2020); GAO et al. (2024); GUU et al. (2020); SHI et al. (2024).

---

## 4. Representações Vetoriais e Embeddings

### 4.1 Conceito

Embeddings são representações numéricas de textos em espaços vetoriais de alta dimensão, capazes de capturar relações semânticas. Textos semanticamente similares produzem vetores próximos no espaço.

### 4.2 Similaridade semântica

A proximidade entre vetores é medida por métricas de distância. Duas métricas principais:

**Similaridade de cosseno:**
$$\text{cos}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|}$$

Valores em $[-1, 1]$; próximo de 1 indica alta similaridade semântica.

**Distância L2 (Euclidiana):**
$$d(\vec{u}, \vec{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

**Este projeto usa distância L2** (configuração padrão do ChromaDB), onde menor distância = maior relevância.

### 4.3 Modelo adotado: BAAI/bge-small-en-v1.5

| Atributo | Valor |
|---|---|
| Dimensões | 384 |
| Parâmetros | 66 milhões |
| Família | BGE (Beijing Academy of AI) |
| Acesso | Local via `sentence-transformers` |

**Critérios de seleção:**
1. **Eficiência computacional** — 66M parâmetros viabilizam inferência local sem GPU
2. **Transferência multilíngue** — apesar do sufixo `-en`, documenta-se transferência robusta para português em benchmarks de recuperação
3. **Independência de API** — a fase de indexação não consome cota da API Google

### 4.4 Banco de dados vetorial: ChromaDB

Banco de dados especializado em armazenamento e consulta eficiente de embeddings. Características relevantes:

- Backend **SQLite3** — persistência local em `storage/`
- Coleção `institucional_db` com **455 vetores** (384 dimensões cada)
- Busca por aproximação de vizinhos mais próximos (ANN) com distância L2
- API Python nativa, sem dependência de servidor externo

**Referências:** arquitetura BGE (BAAI); ChromaDB documentation.

---

## 5. Busca Híbrida em Sistemas de Recuperação de Informação

### 5.1 Limitações da busca semântica pura

A recuperação densa (embeddings) captura similaridade semântica, mas pode falhar em domínios especializados onde **termos técnicos exatos** precisam ser correspondidos — siglas, códigos de disciplinas, nomes de regulamentos específicos.

A recuperação esparsa (BM25) captura correspondência léxica exata, mas ignora sinônimos e variações semânticas.

**Solução:** combinar os dois métodos por busca híbrida.

### 5.2 BM25 — Okapi BM25 e variante Lucene (BM25+)

Fundamentado no modelo probabilístico de relevância (ROBERTSON; ZARAGOZA, 2009), o BM25 calcula a relevância de um documento dado uma consulta com base na frequência dos termos.

**Este projeto usa a variante BM25+** (Lucene), implementada por `bm25s` 0.3.6:

$$\text{score}(d, q) = \sum_{i=1}^{n} \text{IDF}(t_i) \cdot \left( \delta + \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} \right)$$

onde:
- $f(t_i, d)$ = frequência do termo $t_i$ no documento $d$
- $|d|$ = comprimento do documento; $\text{avgdl}$ = comprimento médio do corpus
- $\delta = 0.5$ = fator de piso da variante BM25+ (garante score mínimo positivo)

**Hiperparâmetros adotados (padrão da biblioteca):**

| Parâmetro | Valor | Descrição |
|---|---|---|
| $k_1$ | 1.5 | Saturação de frequência de termos |
| $b$ | 0.75 | Normalização pelo comprimento do documento |
| $\delta$ | 0.5 | Fator de piso (diferencial BM25+) |

### 5.3 Reciprocal Rank Fusion (RRF)

Proposto por Cormack et al. (2009), o RRF combina múltiplos rankings independentes sem depender da escala absoluta dos scores — fundamental quando se fundem scores de naturezas distintas (L2 e BM25 não são comparáveis diretamente).

A pontuação RRF de um documento $d$ é:

$$\text{score}_{\text{RRF}}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

onde:
- $R$ = conjunto de rankings (vetorial e BM25)
- $\text{rank}_r(d)$ = posição do documento $d$ no ranking $r$ (indexado a partir de zero)
- $k = 60$ = constante de suavização padrão da literatura

**Propriedade principal:** um documento presente nos primeiros lugares de **ambos** os rankings acumula score superior ao de um documento em destaque em apenas um ranking.

**Escala dos scores RRF:** ficam na faixa $[0.010, 0.033]$ — incompatível com filtros de similaridade projetados para cosine similarity $[0, 1]$.

**Implementação:** `core/retriever.py`, classe `HybridRetriever`, método `_reciprocal_rank_fusion()`.

**Referências:** ROBERTSON; ZARAGOZA (2009); Cormack et al. (2009); `bm25s` documentation.

---

## 6. Avaliação de Sistemas RAG — Framework RAGAS

### 6.1 Paradigma LLM-as-a-Judge

O RAGAS (Es et al., 2024) adota o paradigma de avaliação automática por LLM juiz, eliminando a necessidade de anotadores humanos para cada resposta avaliada. Um modelo de linguagem (neste projeto: Gemini 2.5 Flash, $T=0.0$) avalia matematicamente as respostas geradas por outro sistema.

### 6.2 Métricas implementadas

#### Faithfulness (Fidelidade)

Mede se as afirmações da resposta são sustentadas pelo contexto recuperado, detectando alucinações.

$$\text{Faithfulness} = \frac{|\text{afirmações suportadas pelo contexto}|}{|\text{total de afirmações na resposta}|}$$

- Valores em $[0, 1]$; próximo de 1 = baixa alucinação
- **Não requer ground truth** — avaliação apenas entre resposta e contexto

#### Answer Relevancy (Relevância da Resposta)

Mede o alinhamento semântico entre a resposta e a pergunta original.

$$\text{AnswerRelevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(\vec{q}, \vec{q}_i^{\,\text{sintética}})$$

O LLM juiz gera $N$ perguntas sintéticas a partir da resposta; o score é a similaridade de cosseno média entre a pergunta original e as sintéticas.

#### Context Precision (Precisão do Contexto)

Mede se os chunks relevantes estão posicionados nas primeiras posições do ranking recuperado.

$$\text{ContextPrecision@k} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\sum_{k=1}^{K} v_k}$$

onde $v_k = 1$ se o $k$-ésimo chunk é relevante (segundo o ground truth), $v_k = 0$ caso contrário.

- **Requer ground truth** — penaliza rankings que colocam chunks irrelevantes antes dos relevantes
- **Diretamente afetada pelo RRF:** melhor fusão → melhor ordenação → maior Context Precision

#### Context Recall (Cobertura do Contexto)

Mede se o conjunto de chunks recuperados cobre todas as informações necessárias para responder.

$$\text{ContextRecall} = \frac{|\text{sentenças do ground truth suportadas pelo contexto}|}{|\text{total de sentenças do ground truth}|}$$

- **Complementar ao Context Precision:** Precision mede qualidade da ordenação; Recall mede completude da recuperação

### 6.3 Relação entre métricas e componentes do sistema

| Métrica | Componente avaliado | Impacto esperado do RRF |
|---|---|---|
| Faithfulness | LLM de síntese (prompt) | Neutro (independente do retriever) |
| Answer Relevancy | LLM de síntese + prompt | Neutro |
| Context Precision | Retriever (ordenação) | ↑ (RRF reordena por relevância combinada) |
| Context Recall | Retriever (cobertura) | ↑ (BM25 complementa gaps semânticos do vetorial) |

**Referências:** ES et al. (2024).

---

## 7. Modelos de Linguagem com Contexto Extendido

### 7.1 Janela de contexto e o problema "Lost in the Middle"

Liu et al. (2024) documentam que LLMs tendem a ignorar informações posicionadas no meio de contextos longos, privilegiando o início e o fim da janela de contexto. Esse fenômeno (*lost in the middle*) penaliza sistemas RAG que injetam muitos chunks no prompt.

**Mitigação adotada:** modo `TREE_SUMMARIZE` do LlamaIndex — agrupa chunks em subconjuntos menores, sumariza cada grupo hierarquicamente e combina os resultados, evitando janelas de contexto excessivamente longas.

### 7.2 Google Gemini 2.5 Flash

| Atributo | Especificação |
|---|---|
| Identificador API | `models/gemini-2.5-flash` |
| Janela de contexto | ~1 milhão de tokens |
| Temperatura de síntese | $T = 0.1$ (respostas determinísticas) |
| Temperatura do LLM juiz | $T = 0.0$ (avaliação totalmente determinística) |
| Acesso | API REST Google; SDK `google-generativeai` |

**Justificativa:** o Gemini 2.5 Flash apresenta baixa latência e alta eficiência em tarefas de raciocínio rápido com grandes volumes de contexto, sendo adequado para síntese de múltiplos chunks documentais em tempo real (GOOGLE DEEPMIND, 2024).

### 7.3 LlamaIndex como orquestrador

O LlamaIndex (versão 0.14.21) orquestra o pipeline RAG completo:

| Componente LlamaIndex | Papel no sistema |
|---|---|
| `SimpleDirectoryReader` | Leitura de PDFs com extração de metadados |
| `SentenceSplitter` | Chunking com controle de sobreposição |
| `VectorStoreIndex` | Índice sobre ChromaDB |
| `VectorIndexRetriever` | Busca semântica top-k |
| `BM25Retriever` | Busca lexical top-k |
| `RetrieverQueryEngine` | Orquestra retriever + synthesizer |
| `TREE_SUMMARIZE` | Síntese hierárquica dos chunks |
| `PromptTemplate` | Template customizado de síntese |

**Referências:** LIU et al. (2024); Google DeepMind (2024); LlamaIndex documentation.

---

## 8. Referências

BROWN, Tom et al. Language Models are Few-Shot Learners. *NeurIPS*, 2020.

CHEN, Yanhan et al. Artificial Intelligence Methods in Natural Language Processing: A Comprehensive Review. *Highlights in Science, Engineering and Technology*, 2024.

ES, Shahul et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. *Proceedings of EACL System Demonstrations*, p. 150–158, 2024.

GAO, Yunfan et al. Retrieval-Augmented Generation for Large Language Models: A Survey. 2024.

GUU, Kelvin et al. REALM: Retrieval-Augmented Language Model Pre-Training. 2020.

JI, Ziwei et al. Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 2023.

JURAFSKY, Daniel; MARTIN, James H. *Speech and Language Processing*. 3. ed. 2019.

KAPLAN, Jared et al. Scaling Laws for Neural Language Models. 2020.

LEWIS, Patrick et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*, 2020.

LIU, Nelson F. et al. Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*, v. 12, p. 157–173, 2024.

QIN, Libo et al. Large Language Models Meet NLP: A Survey. *Frontiers of Computer Science*, 2024.

ROBERTSON, Stephen; ZARAGOZA, Hugo. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, 2009.

SHI, Weijia et al. REPLUG: Retrieval-Augmented Black-Box Language Models. 2024.

WEI, Jason et al. Emergent Abilities of Large Language Models. *Transactions on Machine Learning Research*, 2022.

YANG, Jingfeng et al. Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond. *ACM Transactions on Knowledge Discovery from Data*, v. 18, n. 6, 2024.

ZHAO, Wayne Xin et al. A Survey of Large Language Models. 2024.
