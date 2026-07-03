# Resultados e Discussão

## 1. Visão Geral

Este capítulo apresenta e discute os resultados da avaliação experimental do sistema Secretário Bot FCT/UFPA, um assistente conversacional baseado em RAG (*Retrieval-Augmented Generation*) para o domínio acadêmico institucional da FCT/UFPA. A avaliação consiste em uma comparação controlada entre as três estratégias de recuperação implementadas — vetorial, lexical (BM25) e híbrida (RRF).

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

## 3. Resultados da Avaliação Comparativa

A avaliação comparativa foi executada com 20 questões por configuração sobre o mesmo conjunto de teste, com o comando `python -m core.evaluator`.

**Tabela 1 — Resultados Comparativos (20 questões por modo)**

| Configuração | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| `hybrid` (RRF) | 0.937 | **0.783** | 0.377 | **0.825** |
| `vector` | **0.948** | 0.732 | 0.342 | 0.625 |
| `bm25` | 0.938 | 0.774 | **0.581** | 0.650 |

---

## 4. Análise por Métrica

### 4.1 Faithfulness (Fidelidade ao Contexto)

Todos os três modos apresentaram Faithfulness elevada e estatisticamente próxima: 0.937 (hybrid), 0.948 (vector) e 0.938 (bm25). Essa convergência era esperada: a métrica avalia se as afirmações da resposta são sustentadas pelo contexto recuperado, sendo primariamente determinada pelo LLM de síntese (Gemini 2.5 Flash) e pelo prompt de geração — componentes compartilhados pelos três modos.

A margem de 1,1 pp a favor do modo vetorial (0.948 vs. 0.937) pode refletir que os chunks recuperados por similaridade semântica tendem a ser mais homogêneos e diretamente alinhados à pergunta, reduzindo a probabilidade de o LLM extrapolar o contexto. Contudo, essa diferença é pequena demais para ser considerada conclusiva.

### 4.2 Answer Relevancy (Relevância da Resposta)

O modo `hybrid` obteve o maior Answer Relevancy (0.783), seguido de `bm25` (0.774) e `vector` (0.732). A diferença de 5,1 pp entre hybrid e vector indica que as respostas geradas a partir do contexto híbrido são mais alinhadas à intenção das perguntas.

Essa superioridade se explica pela complementaridade das duas estratégias de recuperação: o componente vetorial captura a intenção semântica da pergunta, enquanto o BM25 garante que termos técnicos específicos (e.g., nomes de disciplinas, siglas, valores numéricos) estejam presentes no contexto. A síntese sobre esse contexto mais rico produz respostas que abordam melhor o que foi perguntado.

### 4.3 Context Precision (Precisão do Contexto)

O resultado mais surpreendente da avaliação: o modo `bm25` obteve Context Precision de **0.581**, superando significativamente `hybrid` (0.377) e `vector` (0.342).

A hipótese inicial era que o RRF melhoraria a precisão ao combinar evidências semânticas e lexicais, posicionando os chunks mais relevantes no topo do ranking. Na prática, o efeito oposto foi observado para esta métrica.

**Interpretação:** O corpus da FCT/UFPA é caracterizado por terminologia técnica padronizada e bem delimitada — nomes de disciplinas, cargas horárias, siglas, artigos de regulamento. Para este tipo de conteúdo, a correspondência exata por palavras-chave (BM25) posiciona os chunks mais relevantes com maior precisão do que a fusão por RRF, que dilui o sinal lexical ao combiná-lo com a componente semântica vetorial. Em outras palavras, quando a resposta está em um documento que contém exatamente os termos da pergunta, o BM25 puro tende a ranquear esse documento em primeiro lugar, enquanto o RRF pode redistribuir os ranks ao dar peso ao retriever vetorial, que pode trazer chunks semanticamente relacionados mas não diretamente responsivos.

### 4.4 Context Recall (Cobertura do Contexto)

O modo `hybrid` liderou com ampla margem em Context Recall (**0.825**), superando `bm25` (0.650) e `vector` (0.625). Essa diferença de 17,5 pp em relação ao segundo colocado é a maior observada entre os modos em qualquer métrica.

Esse resultado confirma a hipótese central do trabalho: ao combinar as duas estratégias de recuperação, o modo híbrido consegue cobrir um conjunto mais amplo de evidências necessárias para responder à pergunta. O componente BM25 recupera chunks com os termos exatos, enquanto o componente vetorial captura chunks semanticamente relacionados que não compartilham os mesmos termos — juntos, eles aumentam a probabilidade de o contexto incluir todas as informações presentes no *ground truth*.

---

## 5. Visão Integrada: Trade-off Precisão × Recall

### 5.1 Por que esta análise é central para o projeto

A comparação entre os três modos de recuperação não é meramente acadêmica — ela fundamenta diretamente a decisão de qual estratégia adotar em produção para o Secretário Bot. Cada modo representa uma filosofia distinta de recuperação de informação:

- **`vector`** converte a pergunta em um vetor de 384 dimensões (modelo `BAAI/bge-small-en-v1.5`) e busca os chunks mais próximos no espaço semântico por distância L2 no ChromaDB. Recupera documentos com *significado parecido*, mesmo sem compartilhar as mesmas palavras — útil para paráfrases e variações de linguagem.

- **`bm25`** aplica o algoritmo BM25+ Lucene sobre os tokens da pergunta e dos documentos, pontuando chunks pela frequência e raridade dos termos coincidentes (k₁=1.5, b=0.75, δ=0.5). Recupera documentos que *contêm exatamente os mesmos termos* da pergunta — eficaz quando o usuário usa a terminologia do documento.

- **`hybrid`** não escolhe entre os dois: executa ambas as buscas (top-20 cada) e combina os rankings via *Reciprocal Rank Fusion* (RRF, k=60), onde cada chunk recebe score `1/(60 + rank)` em cada lista e os scores são somados. O resultado é um ranking único que equilibra evidência semântica e lexical.

A comparação entre esses modos é essencial para responder à pergunta de pesquisa do TCC: *a estratégia híbrida produz resultados superiores às abordagens individuais em um domínio acadêmico de corpus fechado?*

### 5.2 Resultados por dimensão

A tabela abaixo sintetiza qual modo venceu em cada dimensão avaliada:

| Dimensão | Melhor modo | Valor | Segundo lugar | Diferença |
|---|---|---|---|---|
| Fidelidade ao contexto (Faithfulness) | `vector` | **0.948** | `bm25` (0.938) | +1,1 pp |
| Relevância da resposta (Answer Relevancy) | `hybrid` | **0.783** | `bm25` (0.774) | +0,9 pp |
| Precisão do contexto (Context Precision) | `bm25` | **0.581** | `hybrid` (0.377) | +20,4 pp |
| Cobertura do contexto (Context Recall) | `hybrid` | **0.825** | `bm25` (0.650) | +17,5 pp |

### 5.3 Interpretação do trade-off

Os resultados expõem um trade-off estrutural entre precisão e cobertura:

**O `bm25` é o mais preciso, mas o menos abrangente.** Com Context Precision de 0.581, o BM25 posiciona os chunks mais relevantes nas primeiras posições do ranking com maior consistência que os demais modos. Isso ocorre porque o corpus da FCT/UFPA possui terminologia técnica padronizada — nomes de disciplinas, siglas, artigos de regulamento — e quando o usuário usa exatamente esses termos, a correspondência lexical é direta e precisa. Porém, seu Context Recall (0.650) é o menor entre os modos com RRF e similar ao vetorial, o que significa que, embora os primeiros chunks sejam os mais relevantes, o conjunto recuperado não cobre todas as informações necessárias para responder completamente.

**O `vector` apresenta o pior desempenho geral em recuperação.** Context Precision (0.342) e Context Recall (0.625) são os menores valores entre os três modos, apesar da Faithfulness levemente superior. Isso sugere que o modelo de embeddings `BAAI/bge-small-en-v1.5`, treinado predominantemente em inglês, não captura com precisão suficiente a estrutura semântica de documentos técnicos em português — limitação discutida na Seção 7.3.

**O `hybrid` equilibra os dois extremos e vence no que mais importa para o usuário.** Ao combinar os rankings do BM25 e do vetorial via RRF, o modo híbrido obtém o maior Context Recall (0.825) — 17,5 pp acima do BM25 — sem sacrificar a relevância das respostas. O Answer Relevancy de 0.783 confirma que o contexto mais abrangente se traduz em respostas que abordam melhor o que o usuário perguntou.

### 5.4 Implicação para o sistema em produção

Para um assistente acadêmico que recebe perguntas abertas e em linguagem natural de estudantes, **completude é mais crítica que precisão do ranking**. Uma pergunta como *"o que preciso para me matricular em TCC I?"* exige que o contexto cubra todos os requisitos — créditos, aprovações, prazo —, não apenas o chunk mais preciso sobre um deles. Nesse cenário, o Context Recall alto do `hybrid` (0.825) é a métrica que mais protege o usuário de respostas incompletas.

Por isso, o modo `hybrid` foi adotado como configuração padrão do Secretário Bot em produção, configurado em `config/settings.py` como `RETRIEVAL_MODE = "hybrid"`.

---

## 6. Avaliação Qualitativa

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

## 7. Discussão

### 7.1 Sobre a Escolha do Modo Híbrido como Padrão

Os resultados validam a adoção do modo `hybrid` como configuração padrão do sistema em produção. Embora o BM25 apresente Context Precision superior, o hybrid tem Context Recall 26,9% maior (0.825 vs. 0.650) — o que significa que, na prática, as respostas do modo híbrido têm menor risco de omitir informações relevantes. Para o domínio acadêmico, onde perguntas como "quais são todos os requisitos para matrícula em TCC?" demandam completude, o recall é mais crítico que a precisão do ranking.

### 7.2 Sobre a Surpresa do BM25 em Context Precision

O resultado do BM25 em Context Precision (0.581) é o achado mais relevante desta avaliação para a literatura de RAG. Em domínios de conhecimento fechado com terminologia bem delimitada, a correspondência lexical exata pode ser mais precisa que a correspondência semântica vetorial, contrariando a intuição de que embeddings densos sempre superam métodos esparsos.

Esse achado está alinhado com estudos anteriores sobre *domain-specific RAG* (e.g., Zhao et al., 2024; Asai et al., 2023) que demonstram que modelos de embeddings treinados em corpora genéricos (como o BAAI/bge-small-en-v1.5, treinado predominantemente em inglês) podem não capturar adequadamente a estrutura semântica de documentos técnicos em outras línguas, favorecendo o BM25 nesses cenários.

### 7.3 Sobre as Limitações do Conjunto de Avaliação

O conjunto de 20 questões, elaborado manualmente pelo próprio desenvolvedor, constitui a principal limitação metodológica desta avaliação. Um conjunto maior, com anotação independente por múltiplos avaliadores, aumentaria a validade externa dos resultados. Os valores obtidos devem ser interpretados como indicativos da direção das melhorias, não como referências absolutas de desempenho.

### 7.4 Sobre o Impacto das Melhorias Implementadas

As três melhorias arquiteturais implementadas ao longo do trabalho — (i) RRF como estratégia de fusão, (ii) buffer de histórico de conversa e (iii) correção do SimilarityPostprocessor — produziram resultados mensuráveis na avaliação comparativa:

- O modo `hybrid` com RRF apresentou o maior Context Recall (0.825) e Answer Relevancy (0.783) entre todas as configurações, validando a premissa de que a fusão de estratégias melhora a cobertura e relevância das respostas.
- A correção do SimilarityPostprocessor (que filtrava indevidamente resultados do RRF por score absoluto) foi necessária para que os modos hybrid e BM25 operassem corretamente, já que os scores RRF (~0.01–0.03) são incompatíveis com o limiar padrão de 0.3.
- A naturalidade da conversa melhorou qualitativamente com a detecção de saudações e small talk, mesmo sem métrica quantitativa associada.

---

## 8. Conclusões Preliminares

1. O modo `hybrid` com RRF é a melhor configuração geral para o domínio acadêmico da FCT/UFPA, com superior Answer Relevancy (0.783) e Context Recall (0.825).

2. O modo `bm25` apresenta Context Precision superior (0.581), sugerindo que para corpora de domínio fechado com terminologia técnica, a busca lexical posiciona os chunks mais relevantes com maior precisão.

3. A Faithfulness elevada e consistente (≥0.937) em todos os modos indica que o sistema raramente alucina — as respostas são sustentadas pelos documentos recuperados.

4. O sistema demonstrou comportamento adequado em testes qualitativos, respondendo corretamente às perguntas do domínio e tratando adequadamente interações fora do escopo.
