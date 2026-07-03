# Resultados e Discussão

## 1. Visão Geral

Este capítulo apresenta e discute os resultados da avaliação experimental do sistema Secretário Bot FCT/UFPA, um assistente conversacional baseado em RAG (*Retrieval-Augmented Generation*) para o domínio acadêmico institucional da FCT/UFPA. A avaliação foi conduzida em dois momentos distintos: (i) uma avaliação baseline realizada antes das melhorias arquiteturais e (ii) uma avaliação comparativa completa entre as três estratégias de recuperação implementadas — vetorial, lexical (BM25) e híbrida (RRF).

O objetivo central é verificar se a combinação de recuperação vetorial e lexical via *Reciprocal Rank Fusion* (RRF) produz resultados superiores às abordagens individuais, medidos pelas métricas do framework RAGAS.

---

## 2. Configuração da Avaliação

### 2.1 Conjunto de Teste

As 19/20 questões do conjunto de avaliação foram elaboradas manualmente com base nos documentos institucionais da FCT/UFPA, cobrindo três categorias:

| Categoria | Descrição | Qtd. |
|---|---|---|
| Factoides | Resposta numérica ou nominal única (e.g., carga horária, prazo) | 5 |
| Procedimentais | Sequências de passos e prazos (e.g., matrícula em TCC) | 4 |
| Condicionais | Raciocínio com premissas compostas (e.g., requisitos cumulativos) | 6 |
| Listagem | Enumeração de itens (e.g., disciplinas de um bloco) | 5 |

Cada questão possui um *ground truth* textual redigido manualmente, utilizado pelo LLM juiz para calcular ContextPrecision e ContextRecall.

### 2.2 LLM Juiz

O framework RAGAS utilizou o Google Gemini 2.5 Flash como LLM juiz com temperatura $T = 0.0$ (determinístico), assegurando reprodutibilidade na avaliação.

### 2.3 Parâmetros dos Modos de Recuperação

| Parâmetro | Valor |
|---|---|
| Top-k por retriever (vetorial e BM25) | 20 |
| Parâmetro k do RRF | 60 |
| BM25 variante | Lucene (k₁=1.5, b=0.75, δ=0.5) |
| Embeddings | BAAI/bge-small-en-v1.5 (384 dim) |
| Modo de síntese | TREE_SUMMARIZE (Gemini 2.5 Flash, T=0.1) |

---

## 3. Resultados do Baseline

O baseline foi coletado antes das melhorias arquiteturais (commit `8fcf688`), representando o estado inicial do sistema: fusão simples por `node_id` (sem RRF), sem buffer de histórico e com `SimilarityPostprocessor` ativo para todos os modos.

**Tabela 1 — Resultados do Baseline (10 questões válidas de 15)**

| Métrica | Valor |
|---|---|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.724 |
| Context Precision | **0.247** |

O **Context Precision de 0.247** foi o resultado mais crítico: indica que, embora os documentos relevantes fossem recuperados, eles não eram posicionados nas primeiras posições do ranking. Esse resultado motivou diretamente a implementação do RRF como estratégia de reordenação.

> **Nota sobre validade:** 5 das 15 questões falharam por expiração da chave de API durante a coleta, reduzindo o conjunto efetivo a 10 questões. A Faithfulness perfeita (1.000) deve ser interpretada com cautela, possivelmente influenciada pelo tamanho reduzido e pela seleção de questões que tiveram sucesso na execução.

---

## 4. Resultados da Avaliação Comparativa

A avaliação comparativa foi executada com 20 questões por configuração sobre o mesmo conjunto de teste, com o comando `python -m core.evaluator`.

**Tabela 2 — Resultados Comparativos (20 questões por modo)**

| Configuração | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| `hybrid` (RRF) | 0.937 | **0.783** | 0.377 | **0.825** |
| `vector` | **0.948** | 0.732 | 0.342 | 0.625 |
| `bm25` | 0.938 | 0.774 | **0.581** | 0.650 |

---

## 5. Análise por Métrica

### 5.1 Faithfulness (Fidelidade ao Contexto)

Todos os três modos apresentaram Faithfulness elevada e estatisticamente próxima: 0.937 (hybrid), 0.948 (vector) e 0.938 (bm25). Essa convergência era esperada: a métrica avalia se as afirmações da resposta são sustentadas pelo contexto recuperado, sendo primariamente determinada pelo LLM de síntese (Gemini 2.5 Flash) e pelo prompt de geração — componentes compartilhados pelos três modos.

A margem de 1,1 pp a favor do modo vetorial (0.948 vs. 0.937) pode refletir que os chunks recuperados por similaridade semântica tendem a ser mais homogêneos e diretamente alinhados à pergunta, reduzindo a probabilidade de o LLM extrapolar o contexto. Contudo, essa diferença é pequena demais para ser considerada conclusiva.

### 5.2 Answer Relevancy (Relevância da Resposta)

O modo `hybrid` obteve o maior Answer Relevancy (0.783), seguido de `bm25` (0.774) e `vector` (0.732). A diferença de 5,1 pp entre hybrid e vector indica que as respostas geradas a partir do contexto híbrido são mais alinhadas à intenção das perguntas.

Essa superioridade se explica pela complementaridade das duas estratégias de recuperação: o componente vetorial captura a intenção semântica da pergunta, enquanto o BM25 garante que termos técnicos específicos (e.g., nomes de disciplinas, siglas, valores numéricos) estejam presentes no contexto. A síntese sobre esse contexto mais rico produz respostas que abordam melhor o que foi perguntado.

### 5.3 Context Precision (Precisão do Contexto)

O resultado mais surpreendente da avaliação: o modo `bm25` obteve Context Precision de **0.581**, superando significativamente `hybrid` (0.377) e `vector` (0.342).

A hipótese inicial era que o RRF melhoraria a precisão ao combinar evidências semânticas e lexicais, posicionando os chunks mais relevantes no topo do ranking. Na prática, o efeito oposto foi observado para esta métrica.

**Interpretação:** O corpus da FCT/UFPA é caracterizado por terminologia técnica padronizada e bem delimitada — nomes de disciplinas, cargas horárias, siglas, artigos de regulamento. Para este tipo de conteúdo, a correspondência exata por palavras-chave (BM25) posiciona os chunks mais relevantes com maior precisão do que a fusão por RRF, que dilui o sinal lexical ao combiná-lo com a componente semântica vetorial. Em outras palavras, quando a resposta está em um documento que contém exatamente os termos da pergunta, o BM25 puro tende a ranquear esse documento em primeiro lugar, enquanto o RRF pode redistribuir os ranks ao dar peso ao retriever vetorial, que pode trazer chunks semanticamente relacionados mas não diretamente responsivos.

Em relação ao baseline (Context Precision = 0.247), todos os três modos apresentaram melhora:
- `bm25`: 0.247 → 0.581 (+135%)
- `hybrid`: 0.247 → 0.377 (+53%)
- `vector`: 0.247 → 0.342 (+39%)

A melhora do modo `hybrid` sobre o baseline confirma que o RRF contribuiu positivamente para a ordenação dos chunks — mas o BM25 contribuiu ainda mais.

### 5.4 Context Recall (Cobertura do Contexto)

O modo `hybrid` liderou com ampla margem em Context Recall (**0.825**), superando `bm25` (0.650) e `vector` (0.625). Essa diferença de 17,5 pp em relação ao segundo colocado é a maior observada entre os modos em qualquer métrica.

Esse resultado confirma a hipótese central do trabalho: ao combinar as duas estratégias de recuperação, o modo híbrido consegue cobrir um conjunto mais amplo de evidências necessárias para responder à pergunta. O componente BM25 recupera chunks com os termos exatos, enquanto o componente vetorial captura chunks semanticamente relacionados que não compartilham os mesmos termos — juntos, eles aumentam a probabilidade de o contexto incluir todas as informações presentes no *ground truth*.

---

## 6. Visão Integrada: Trade-off Precisão × Recall

Os resultados revelam um trade-off claro entre as estratégias:

| Dimensão | Melhor modo | Valor |
|---|---|---|
| Fidelidade ao contexto | `vector` | 0.948 |
| Relevância da resposta | `hybrid` | 0.783 |
| Precisão do contexto | `bm25` | 0.581 |
| Cobertura do contexto | `hybrid` | 0.825 |

O modo `hybrid` vence nas duas métricas de maior impacto direto na experiência do usuário: **Answer Relevancy** (quão bem a resposta aborda a pergunta) e **Context Recall** (quão completo é o contexto). O modo `bm25` é superior em Context Precision mas inferior em Recall, enquanto o modo `vector` apresenta o pior desempenho geral em relevância e recall.

Considerando que o sistema é projetado para **usuários finais que fazem perguntas naturais sobre documentos acadêmicos**, a métrica de maior impacto percebido é a Answer Relevancy — e para ela, o modo `hybrid` é consistentemente superior.

---

## 7. Comparação com o Baseline

**Tabela 3 — Comparação Baseline × Avaliação Comparativa**

| Métrica | Baseline | Hybrid | Vector | BM25 |
|---|---|---|---|---|
| Faithfulness | 1.000* | 0.937 | 0.948 | 0.938 |
| Answer Relevancy | 0.724 | **0.783** (+8,2%) | 0.732 (+1,1%) | 0.774 (+6,9%) |
| Context Precision | 0.247 | 0.377 (+53%) | 0.342 (+39%) | **0.581 (+135%)** |
| Context Recall | — | **0.825** | 0.625 | 0.650 |

> *Faithfulness do baseline de 1.000 deve ser interpretada com cautela (10 questões, possível viés de seleção).

Destaca-se que:
1. O Answer Relevancy melhorou em todos os modos comparado ao baseline (+1,1% a +8,2%), com o hybrid apresentando a maior evolução.
2. O Context Precision melhorou expressivamente em todos os modos, confirmando que as melhorias arquiteturais (RRF + correção do SimilarityPostprocessor) impactaram positivamente a ordenação dos chunks.
3. O Context Recall não foi coletado no baseline; os valores da avaliação comparativa servem como referência inicial.

---

## 8. Avaliação Qualitativa

Além da avaliação quantitativa, o sistema foi testado qualitativamente através de conversas diretas pelo Telegram. Os principais comportamentos observados:

**Comportamentos corretos:**
- Resposta correta e completa às perguntas sobre ementas, cargas horárias e blocos curriculares
- Citação do documento de origem ao final de cada resposta
- Detecção e resposta adequada a saudações (incluindo gírias: "aoba", "tmj", "oi")
- Redirecionamento educado para perguntas fora do escopo (small talk, perguntas gerais)
- Manutenção de contexto conversacional por até 5 turnos (buffer de histórico)

**Limitações observadas:**
- Perguntas sobre informações não presentes no corpus geram respostas de "não encontrado" adequadas, mas sem sugestão de fontes alternativas
- Perguntas altamente ambíguas podem recuperar chunks de múltiplos documentos, produzindo respostas longas com informações de diferentes contextos
- Documentos com formatação complexa (tabelas em PDF escaneado) podem ter sido indexados com perda de estrutura tabular

---

## 9. Discussão

### 9.1 Sobre a Escolha do Modo Híbrido como Padrão

Os resultados validam a adoção do modo `hybrid` como configuração padrão do sistema em produção. Embora o BM25 apresente Context Precision superior, o hybrid tem Context Recall 26,9% maior (0.825 vs. 0.650) — o que significa que, na prática, as respostas do modo híbrido têm menor risco de omitir informações relevantes. Para o domínio acadêmico, onde perguntas como "quais são todos os requisitos para matrícula em TCC?" demandam completude, o recall é mais crítico que a precisão do ranking.

### 9.2 Sobre a Surpresa do BM25 em Context Precision

O resultado do BM25 em Context Precision (0.581) é o achado mais relevante desta avaliação para a literatura de RAG. Em domínios de conhecimento fechado com terminologia bem delimitada, a correspondência lexical exata pode ser mais precisa que a correspondência semântica vetorial, contrariando a intuição de que embeddings densos sempre superam métodos esparsos.

Esse achado está alinhado com estudos anteriores sobre *domain-specific RAG* (e.g., Zhao et al., 2024; Asai et al., 2023) que demonstram que modelos de embeddings treinados em corpora genéricos (como o BAAI/bge-small-en-v1.5, treinado predominantemente em inglês) podem não capturar adequadamente a estrutura semântica de documentos técnicos em outras línguas, favorecendo o BM25 nesses cenários.

### 9.3 Sobre as Limitações do Conjunto de Avaliação

O conjunto de 20 questões, elaborado manualmente pelo próprio desenvolvedor, constitui a principal limitação metodológica desta avaliação. Um conjunto maior, com anotação independente por múltiplos avaliadores, aumentaria a validade externa dos resultados. Os valores obtidos devem ser interpretados como indicativos da direção das melhorias, não como referências absolutas de desempenho.

### 9.4 Sobre o Impacto das Melhorias Implementadas

As três melhorias arquiteturais implementadas ao longo do trabalho — (i) RRF como estratégia de fusão, (ii) buffer de histórico de conversa e (iii) correção do SimilarityPostprocessor — produziram ganhos mensuráveis:

- O Context Precision melhorou entre 39% e 135% em relação ao baseline, validando o impacto da correção do postprocessor e do RRF.
- O Answer Relevancy melhorou 8,2% no modo hybrid, indicando que a melhor qualidade do contexto recuperado se traduz em respostas mais relevantes.
- A naturalidade da conversa melhorou qualitativamente com a detecção de saudações e small talk, mesmo sem métrica quantitativa associada.

---

## 10. Conclusões Preliminares

1. O modo `hybrid` com RRF é a melhor configuração geral para o domínio acadêmico da FCT/UFPA, com superior Answer Relevancy (0.783) e Context Recall (0.825).

2. O modo `bm25` apresenta Context Precision superior (0.581), sugerindo que para corpora de domínio fechado com terminologia técnica, a busca lexical posiciona os chunks mais precisamente.

3. Todas as melhorias arquiteturais implementadas produziram ganhos mensuráveis sobre o baseline, especialmente em Context Precision (+39% a +135%).

4. A Faithfulness elevada e consistente (≥0.937) em todos os modos indica que o sistema raramente alucina — as respostas são sustentadas pelos documentos recuperados.

5. O sistema demonstrou comportamento adequado em testes qualitativos, respondendo corretamente às perguntas do domínio e tratando adequadamente interações fora do escopo.
