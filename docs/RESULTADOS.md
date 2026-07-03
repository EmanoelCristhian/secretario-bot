# Resultados e Discussão

## 1. Visão Geral

Este capítulo apresenta e discute os resultados da avaliação experimental do sistema Secretário Bot FCT/UFPA, um assistente conversacional baseado em RAG (*Retrieval-Augmented Generation*) para o domínio acadêmico institucional da FCT/UFPA. A avaliação consiste em uma comparação controlada entre as três estratégias de recuperação implementadas — vetorial, lexical (BM25) e híbrida (RRF).

O objetivo central é responder à pergunta de pesquisa do TCC: **a combinação de recuperação vetorial e lexical via *Reciprocal Rank Fusion* produz resultados superiores às abordagens individuais em um domínio acadêmico de corpus fechado?**

Para responder a essa pergunta, cada estratégia foi avaliada de forma isolada e comparativa sob as quatro métricas do framework RAGAS — Faithfulness, Answer Relevancy, Context Precision e Context Recall —, que avaliam dimensões ortogonais da qualidade de um pipeline RAG: fidelidade ao contexto, relevância da resposta, precisão e cobertura da recuperação.

A importância desta avaliação vai além da validação técnica: ela fornece evidência empírica para embasar decisões arquiteturais em sistemas RAG aplicados a domínios institucionais em português, contexto com escassa literatura experimental específica.

---

## 2. Configuração da Avaliação

### 2.1 Conjunto de Teste

As 20 questões do conjunto de avaliação foram elaboradas manualmente com base nos documentos institucionais da FCT/UFPA. A escolha de construção manual — e não automática via LLM — se justifica pela necessidade de garantir que as perguntas representem dúvidas reais e recorrentes de estudantes, e não variações artificiais geradas a partir dos próprios documentos.

As questões cobrem quatro categorias, cada uma testando um aspecto distinto da capacidade do sistema:

| Categoria | Descrição | Qtd. | Por que incluir |
|---|---|---|---|
| Factoides | Resposta numérica ou nominal única (e.g., carga horária, prazo) | 5 | Testa recuperação precisa de valores específicos |
| Procedimentais | Sequências de passos e prazos (e.g., matrícula em TCC) | 4 | Testa síntese de múltiplos chunks em ordem lógica |
| Condicionais | Raciocínio com premissas compostas (e.g., requisitos cumulativos) | 6 | Testa capacidade do LLM de combinar condições do contexto |
| Listagem | Enumeração de itens (e.g., disciplinas de um bloco) | 5 | Testa completude — o sistema deve listar todos os itens, não apenas alguns |

Cada questão possui um *ground truth* textual redigido manualmente, que representa a resposta esperada com base nos documentos. Esse ground truth é utilizado pelo LLM juiz para calcular Context Precision (os chunks recuperados estão ordenados corretamente?) e Context Recall (o conjunto de chunks cobre tudo o que está no ground truth?).

### 2.2 Por que RAGAS como Framework de Avaliação

O RAGAS (*Retrieval Augmented Generation Assessment*) foi escolhido como framework de avaliação por três razões principais:

1. **Avaliação sem referência de resposta perfeita para todas as métricas:** Faithfulness e Answer Relevancy são calculadas sem depender de uma resposta de referência — o LLM juiz avalia a resposta gerada diretamente contra o contexto recuperado e a pergunta original. Isso reduz o esforço de anotação humana.

2. **Decomposição da qualidade do pipeline:** ao separar a avaliação em quatro métricas, o RAGAS permite identificar onde o sistema falha — se o problema é a recuperação (Precision/Recall baixos), a síntese (Faithfulness baixa) ou o alinhamento da resposta com a pergunta (Answer Relevancy baixo). Isso é essencial para orientar melhorias arquiteturais futuras.

3. **Maturidade e adoção:** o framework é amplamente utilizado em pesquisas sobre RAG (Es et al., 2024), facilitando comparação com trabalhos anteriores.

### 2.3 LLM Juiz

O framework RAGAS utilizou o Google Gemini 2.5 Flash como LLM juiz com temperatura $T = 0.0$ (modo determinístico), assegurando que a mesma pergunta produza sempre a mesma avaliação, tornando os resultados reprodutíveis. O mesmo modelo é usado para geração ($T = 0.1$) — a diferença de temperatura reflete o papel distinto: geração admite variação criativa mínima; avaliação exige consistência absoluta.

### 2.4 Parâmetros dos Modos de Recuperação

Todos os modos compartilham os mesmos parâmetros de indexação e síntese, variando apenas na estratégia de recuperação. Isso garante que as diferenças nas métricas reflitam exclusivamente o impacto do retriever:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Top-k por retriever (vetorial e BM25) | 20 | Equilíbrio entre cobertura e custo de tokens no contexto |
| Parâmetro k do RRF | 60 | Valor padrão da literatura (Cormack et al., 2009); reduz sensibilidade a rankings de baixa posição |
| BM25 variante | Lucene (k₁=1.5, b=0.75, δ=0.5) | Variante com penalidade de frequência suavizada, robusta para documentos de tamanhos variados |
| Embeddings | BAAI/bge-small-en-v1.5 (384 dim) | Modelo compacto (66M parâmetros) executável localmente sem GPU; transferência multilíngue razoável |
| Modo de síntese | TREE_SUMMARIZE (Gemini 2.5 Flash, T=0.1) | Agrega múltiplos chunks hierarquicamente, reduzindo o problema de "lost-in-the-middle" |

---

## 3. Resultados da Avaliação Comparativa

A avaliação comparativa foi executada com 20 questões por configuração sobre o mesmo conjunto de teste, com o comando `python -m core.evaluator`. O script executa os três modos sequencialmente, garantindo que todas as configurações respondam exatamente ao mesmo conjunto de perguntas e *ground truths*.

**Tabela 1 — Resultados Comparativos (20 questões por modo)**

| Configuração | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| `hybrid` (RRF) | 0.937 | **0.783** | 0.377 | **0.825** |
| `vector` | **0.948** | 0.732 | 0.342 | 0.625 |
| `bm25` | 0.938 | 0.774 | **0.581** | 0.650 |

Todas as métricas estão no intervalo $[0, 1]$, onde valores mais próximos de 1 indicam melhor desempenho. Os valores em negrito indicam o melhor resultado por coluna. Nenhum modo dominou em todas as métricas simultaneamente — cada estratégia apresentou vantagem em pelo menos uma dimensão, tornando a análise qualitativa essencial para a tomada de decisão.

---

## 4. Análise por Métrica

### 4.1 Faithfulness (Fidelidade ao Contexto)

**O que mede:** Faithfulness avalia se as afirmações presentes na resposta gerada são sustentadas pelo contexto recuperado. Formalmente, o LLM juiz decompõe a resposta em afirmações atômicas e verifica, para cada uma, se há suporte no contexto:

$$\text{Faithfulness} = \frac{|\text{afirmações suportadas pelo contexto}|}{|\text{total de afirmações na resposta}|}$$

**Por que é importante para este projeto:** Um assistente acadêmico que "inventa" informações — como prazos ou requisitos que não constam nos documentos — causa dano direto ao estudante. Faithfulness é a métrica de segurança do sistema: valores próximos de 1 garantem que o bot não alucina.

**Resultados:** Todos os três modos apresentaram Faithfulness elevada e estatisticamente próxima: 0.937 (hybrid), 0.948 (vector) e 0.938 (bm25). Essa convergência era esperada: Faithfulness é determinada primariamente pelo LLM de síntese (Gemini 2.5 Flash) e pelo prompt de geração — componentes compartilhados pelos três modos. Independentemente de qual estratégia recupera os chunks, o mesmo LLM sintetiza a resposta com a mesma instrução de "não extrapolar o contexto".

**Interpretação da diferença entre modos:** A margem de 1,1 pp a favor do modo vetorial (0.948 vs. 0.937) pode refletir que os chunks recuperados por similaridade semântica tendem a ser mais homogêneos e diretamente alinhados à pergunta, reduzindo a chance de o LLM receber trechos contraditórios ou tangenciais que motivem extrapolações. No modo híbrido, o BM25 pode incluir chunks com correspondência lexical alta mas relevância semântica menor, levemente aumentando a possibilidade de o LLM usar informações de contexto incorreto. Contudo, a diferença de 1,1 pp é pequena demais para ser considerada conclusiva — todos os modos são igualmente seguros contra alucinação.

### 4.2 Answer Relevancy (Relevância da Resposta)

**O que mede:** Answer Relevancy avalia o alinhamento entre a resposta gerada e a intenção da pergunta original, independentemente da veracidade do conteúdo. O LLM juiz gera $N$ perguntas sintéticas a partir da resposta e calcula a similaridade cosseno entre os embeddings dessas perguntas e a pergunta original:

$$\text{AnswerRelevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(\vec{q}, \vec{q}_i^{\,\text{sintética}})$$

Uma resposta que "foge do assunto" ou é excessivamente genérica produz perguntas sintéticas divergentes da pergunta original, resultando em score baixo.

**Por que é importante para este projeto:** Esta é a métrica mais diretamente percebida pelo usuário final. Um estudante que pergunta "qual é a carga horária de Cálculo I?" e recebe uma resposta correta mas que discorre sobre o histórico do curso perceberá a resposta como inadequada, mesmo que seja fiel ao contexto. Answer Relevancy captura exatamente essa percepção.

**Resultados:** O modo `hybrid` obteve o maior Answer Relevancy (0.783), seguido de `bm25` (0.774) e `vector` (0.732). A diferença de 5,1 pp entre hybrid e vector é a maior entre os modos nesta métrica e tem implicação prática direta: respostas geradas no modo híbrido abordam melhor a pergunta do usuário.

**Interpretação:** A superioridade do hybrid se explica pela complementaridade das duas estratégias. O componente vetorial recupera chunks semanticamente alinhados à intenção da pergunta; o BM25 garante que os termos técnicos exatos da pergunta (nomes de disciplinas, siglas, valores numéricos) estejam presentes no contexto. Quando o Gemini sintetiza sobre esse contexto mais rico e preciso, a resposta resultante tende a abordar com mais completude e foco o que foi perguntado. O modo vetorial puro, ao recuperar chunks por semântica sem garantir presença dos termos exatos, produz contextos que podem ser *relacionados* mas não *responsivos*, levando a respostas mais genéricas.

### 4.3 Context Precision (Precisão do Contexto)

**O que mede:** Context Precision avalia se os chunks mais relevantes estão posicionados nas primeiras posições do ranking recuperado. A métrica penaliza quando o sistema recupera o chunk correto, mas o coloca em posição baixa, atrás de chunks menos relevantes:

$$\text{ContextPrecision@k} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\sum_{k=1}^{K} v_k}$$

onde $v_k = 1$ se o $k$-ésimo chunk é relevante segundo o ground truth e $v_k = 0$ caso contrário.

**Por que é importante para este projeto:** O LLM de síntese recebe os chunks na ordem em que foram recuperados. Pesquisas sobre o efeito "lost-in-the-middle" (Liu et al., 2023) demonstram que LLMs tendem a utilizar melhor informações posicionadas no início e no final do contexto, negligenciando o meio. Um sistema com Context Precision baixo coloca as informações mais relevantes no meio do contexto, aumentando o risco de o LLM ignorá-las ou sintetizá-las com menor peso.

**Resultado mais surpreendente da avaliação:** O modo `bm25` obteve Context Precision de **0.581**, superando significativamente `hybrid` (0.377) e `vector` (0.342) — uma diferença de 20,4 pp em relação ao segundo colocado.

A hipótese inicial era que o RRF melhoraria a precisão ao combinar evidências semânticas e lexicais. Na prática, o efeito oposto foi observado para esta métrica específica.

**Interpretação:** O corpus da FCT/UFPA é caracterizado por terminologia técnica padronizada e bem delimitada — nomes de disciplinas, cargas horárias, siglas, artigos de regulamento. Para este tipo de conteúdo, quando o usuário pergunta "qual a carga horária de Algoritmos e Estruturas de Dados?", o documento que contém exatamente esses termos terá score BM25 máximo e será posicionado no topo do ranking. O RRF, ao combinar esse ranking com o vetorial, pode redistribuir as posições ao dar peso a chunks semanticamente próximos mas que não contêm os termos exatos — diluindo o sinal lexical preciso do BM25 e reduzindo a Precision do conjunto combinado. Em domínios com linguagem natural variada, o RRF tende a melhorar a Precision; em domínios técnicos com terminologia exata, o BM25 puro pode ser mais eficaz nessa dimensão.

### 4.4 Context Recall (Cobertura do Contexto)

**O que mede:** Context Recall avalia se o conjunto de chunks recuperados cobre todas as informações necessárias para responder à pergunta, segundo o ground truth:

$$\text{ContextRecall} = \frac{|\text{sentenças do ground truth suportadas pelo contexto}|}{|\text{total de sentenças do ground truth}|}$$

O LLM juiz verifica, para cada sentença do ground truth, se ela pode ser atribuída a algum chunk do contexto recuperado. Recall baixo significa que o sistema não encontrou parte das informações necessárias — independentemente de quão bem-ordenados estejam os chunks que encontrou.

**Por que é importante para este projeto:** Context Recall é especialmente crítico para perguntas que demandam informações distribuídas em múltiplos documentos ou seções. Uma pergunta como "quais são os requisitos para integralização do curso?" pode exigir chunks de documentos diferentes (regulamento, grade curricular, normas de atividades complementares). Um sistema com Recall baixo omite parte dessas fontes, produzindo respostas incompletas que podem induzir o estudante a erro.

**Resultados:** O modo `hybrid` liderou com ampla margem em Context Recall (**0.825**), superando `bm25` (0.650) e `vector` (0.625). A diferença de 17,5 pp em relação ao segundo colocado (bm25) é a maior diferença absoluta observada entre os modos em qualquer métrica da avaliação.

**Interpretação:** Este resultado confirma diretamente a hipótese central do trabalho. O modo híbrido combina dois mecanismos complementares de recuperação: o BM25 recupera chunks que contêm os termos exatos da pergunta; o vetorial recupera chunks semanticamente relacionados que descrevem o mesmo conceito com palavras diferentes. Juntos, eles cobrem uma superfície maior do espaço de informações relevantes. Um chunk que descreve "prazo de entrega do relatório de estágio" sem usar a palavra "prazo" (usando, por exemplo, "data limite") pode ser recuperado pelo vetorial mas não pelo BM25 — e vice-versa. O RRF garante que ambos apareçam no contexto final, aumentando a probabilidade de o ground truth estar coberto.

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

**O `bm25` é o mais preciso, mas o menos abrangente.** Com Context Precision de 0.581, o BM25 posiciona os chunks mais relevantes nas primeiras posições do ranking com maior consistência que os demais modos. Isso ocorre porque o corpus da FCT/UFPA possui terminologia técnica padronizada — nomes de disciplinas, siglas, artigos de regulamento — e quando o usuário usa exatamente esses termos, a correspondência lexical é direta e precisa. Porém, seu Context Recall (0.650) é o menor entre os modos, o que significa que, embora os primeiros chunks sejam os mais relevantes, o conjunto recuperado não cobre todas as informações necessárias para responder completamente.

**O `vector` apresenta o pior desempenho geral em recuperação.** Context Precision (0.342) e Context Recall (0.625) são os menores valores entre os três modos, apesar da Faithfulness levemente superior. Isso sugere que o modelo de embeddings `BAAI/bge-small-en-v1.5`, treinado predominantemente em inglês, não captura com precisão suficiente a estrutura semântica de documentos técnicos em português — limitação discutida na Seção 7.3.

**O `hybrid` equilibra os dois extremos e vence no que mais importa para o usuário.** Ao combinar os rankings do BM25 e do vetorial via RRF, o modo híbrido obtém o maior Context Recall (0.825) — 17,5 pp acima do BM25 — sem sacrificar a relevância das respostas. O Answer Relevancy de 0.783 confirma que o contexto mais abrangente se traduz em respostas que abordam melhor o que o usuário perguntou.

### 5.4 Implicação para o sistema em produção

Para um assistente acadêmico que recebe perguntas abertas e em linguagem natural de estudantes, **completude é mais crítica que precisão do ranking**. Uma pergunta como *"o que preciso para me matricular em TCC I?"* exige que o contexto cubra todos os requisitos — créditos, aprovações, prazo —, não apenas o chunk mais preciso sobre um deles. Nesse cenário, o Context Recall alto do `hybrid` (0.825) é a métrica que mais protege o usuário de respostas incompletas.

Por isso, o modo `hybrid` foi adotado como configuração padrão do Secretário Bot em produção, configurado em `config/settings.py` como `RETRIEVAL_MODE = "hybrid"`.

---

## 6. Avaliação Qualitativa

### 6.1 Por que a avaliação qualitativa complementa o RAGAS

As métricas RAGAS avaliam a qualidade do pipeline RAG de forma objetiva, mas não capturam todos os aspectos relevantes da experiência do usuário. Um sistema pode ter Answer Relevancy alta e ainda assim falhar em interações não-documentais — saudações, mensagens fora do escopo, erros de digitação, perguntas ambíguas. A avaliação qualitativa, conduzida por meio de conversas diretas no Telegram, complementa o RAGAS ao verificar comportamentos que os benchmarks não cobrem.

### 6.2 Comportamentos corretos observados

- **Respostas corretas e completas sobre ementas e grades:** Perguntas como "quais disciplinas fazem parte do 3º bloco de Engenharia da Computação?" foram respondidas com listagem completa e citação do documento de origem, confirmando que o Context Recall alto (0.825 no modo hybrid) se traduz em qualidade perceptível.

- **Citação do documento de origem:** Ao final de cada resposta, o sistema indica o arquivo PDF de onde a informação foi extraída (e.g., `ementas_bloco_iii_engcomp.pdf`), permitindo ao estudante verificar a fonte diretamente.

- **Detecção de saudações e gírias brasileiras:** O `GreetingDetector` classifica corretamente saudações formais ("olá", "boa tarde") e informais ("oi", "aoba", "tmj", "iáe"), respondendo com uma mensagem de boas-vindas ao invés de acionar o pipeline RAG desnecessariamente.

- **Redirecionamento educado para perguntas fora do escopo:** Mensagens de small talk (e.g., "que dia bom hoje!") ou perguntas gerais ("me explica o que é machine learning") são identificadas pelo `GreetingDetector` e respondidas com redirecionamento gentil para o domínio do sistema, sem invocar o LLM de síntese.

- **Manutenção de contexto conversacional:** O buffer de histórico (deque de 5 turnos) permite que perguntas de acompanhamento façam sentido sem repetir o contexto. Por exemplo: primeiro turno "quais as disciplinas do bloco I?", segundo turno "e as do bloco II?" — o sistema compreende a continuidade.

- **Interceptação de respostas em inglês do LlamaIndex:** O `ResponseValidator` detecta e substitui mensagens de fallback em inglês geradas internamente pelo LlamaIndex (e.g., "The provided context does not contain...") por mensagens em português com orientação ao usuário.

### 6.3 Limitações observadas

- **Perguntas sobre informações ausentes no corpus:** Quando o estudante pergunta sobre algo não coberto pelos 31 PDFs (e.g., horário de atendimento da secretaria, calendário acadêmico), o sistema responde corretamente que não encontrou a informação, mas não sugere onde buscar — limitação que poderia ser endereçada com integração a fontes externas.

- **Perguntas altamente ambíguas:** Consultas sem contexto suficiente (e.g., "quantos créditos?") podem recuperar chunks de múltiplos documentos e gerar respostas longas cobrindo diferentes cursos, o que pode confundir o usuário. A adição de uma etapa de clarificação de intenção poderia mitigar esse comportamento.

- **Perda de estrutura tabular em PDFs:** Documentos com tabelas complexas (e.g., grades curriculares com células mescladas) são processados pelo `SimpleDirectoryReader` como texto plano, podendo perder a relação entre colunas e linhas durante a extração. Isso pode resultar em chunks que listam disciplinas sem associá-las corretamente às suas cargas horárias.

---

## 7. Discussão

### 7.1 Sobre a Escolha do Modo Híbrido como Padrão

Os resultados validam a adoção do modo `hybrid` como configuração padrão do sistema em produção. A decisão se baseia na priorização do Context Recall (0.825) e do Answer Relevancy (0.783) — as duas métricas de maior impacto direto na experiência do estudante.

Embora o BM25 apresente Context Precision superior (0.581 vs. 0.377), o hybrid tem Context Recall 26,9% maior (0.825 vs. 0.650). Na prática, isso significa que o modo híbrido tem menor risco de omitir informações relevantes na resposta. Para perguntas acadêmicas que tipicamente envolvem múltiplos requisitos ou condições (e.g., "quando posso me matricular em TCC?", "quais são os pré-requisitos de Sistemas Operacionais?"), omitir um requisito é pior do que rankear os chunks em ordem subótima — o estudante pode tomar uma decisão incorreta com base em uma resposta parcialmente incompleta.

### 7.2 Sobre a Surpresa do BM25 em Context Precision

O resultado do BM25 em Context Precision (0.581) é o achado mais relevante desta avaliação para a literatura de RAG aplicado a domínios institucionais. Em domínios de conhecimento fechado com terminologia bem delimitada, a correspondência lexical exata pode ser mais precisa que a correspondência semântica vetorial, contrariando a intuição de que embeddings densos sempre superam métodos esparsos.

Esse achado se alinha com estudos sobre *domain-specific RAG* que demonstram que modelos de embeddings treinados em corpora genéricos (como o BAAI/bge-small-en-v1.5, predominantemente em inglês) podem não capturar adequadamente a estrutura semântica de documentos técnicos em outros idiomas. Quando o vocabulário do domínio é específico e pouco representado nos dados de treinamento do modelo de embeddings, o espaço vetorial resultante pode não separar adequadamente conceitos similares, e a busca por distância L2 perde eficácia relativa ao BM25.

Uma implicação prática: para corpora institucionais em português com terminologia técnica, o BM25 deve ser considerado como retriever primário em cenários onde a precisão do ranking é crítica (e.g., sistemas onde apenas o primeiro resultado é exibido ao usuário, sem síntese por LLM).

### 7.3 Sobre as Limitações do Conjunto de Avaliação

O conjunto de 20 questões, elaborado manualmente pelo próprio desenvolvedor, constitui a principal limitação metodológica desta avaliação. Três aspectos merecem atenção:

**Tamanho:** 20 questões é um conjunto pequeno para avaliação robusta. Variações na seleção de questões podem afetar os valores médios de forma significativa. Um conjunto de 100+ questões, como recomendado pela literatura (Es et al., 2024), aumentaria a estabilidade estatística dos resultados.

**Viés de seleção:** As perguntas foram elaboradas pelo mesmo desenvolvedor que construiu o sistema, o que pode introduzir viés inconsciente de seleção — perguntas para as quais o sistema foi implicitamente otimizado. Um processo de anotação independente por estudantes reais da FCT/UFPA aumentaria a validade externa.

**Ground truth subjetivo:** Para perguntas procedimentais e condicionais, a redação do ground truth envolve decisões editoriais (nível de detalhe, ordem de apresentação) que podem influenciar os scores de Recall. Dois anotadores diferentes podem produzir ground truths ligeiramente distintos para a mesma pergunta.

Os valores obtidos devem ser interpretados como indicativos da direção das diferenças entre os modos, não como referências absolutas de desempenho do sistema.

### 7.4 Sobre o Impacto das Melhorias Implementadas

Três melhorias arquiteturais foram implementadas ao longo do desenvolvimento, e os resultados da avaliação comparativa permitem estimar seu impacto:

**RRF como estratégia de fusão:** A implementação do Reciprocal Rank Fusion substituiu uma fusão simples por `node_id` que apenas deduplicava chunks sem reordenar. O RRF produziu o maior Context Recall (0.825 no modo hybrid) e o maior Answer Relevancy (0.783), validando a premissa de que a fusão ponderada por posição supera a deduplicação simples. O parâmetro k=60 foi mantido no valor padrão da literatura, mas ajustes finos poderiam explorar se valores menores (maior peso às primeiras posições) beneficiariam a Context Precision do modo híbrido.

**Correção do SimilarityPostprocessor:** O postprocessor original filtrava chunks com score de similaridade abaixo de 0.3, um limiar adequado para scores de distância cosseno mas incompatível com os scores RRF, que variam tipicamente entre 0.01 e 0.03. Sem essa correção, os modos hybrid e BM25 descartavam a maioria dos chunks recuperados, produzindo contextos vazios ou insuficientes. A correção — desativando o postprocessor para esses modos — foi necessária para que os resultados reportados fossem tecnicamente válidos.

**Buffer de histórico de conversa:** O buffer de 5 turnos (implementado via `deque` em `utils/conversation_history.py`) não tem métrica RAGAS correspondente, pois o conjunto de avaliação consiste em perguntas independentes. Seu impacto foi verificado qualitativamente: perguntas de acompanhamento são compreendidas corretamente sem repetição de contexto pelo usuário.

---

## 8. Conclusões Preliminares

1. **O modo `hybrid` com RRF é a melhor configuração geral** para o domínio acadêmico da FCT/UFPA, com superior Answer Relevancy (0.783) e Context Recall (0.825). A combinação das duas estratégias de recuperação produz respostas mais completas e mais alinhadas à intenção do usuário do que qualquer estratégia individual.

2. **O modo `bm25` surpreende em Context Precision (0.581)**, sugerindo que, para corpora de domínio fechado com terminologia técnica padronizada em português, a correspondência lexical exata posiciona os chunks mais relevantes com maior precisão do que embeddings densos treinados predominantemente em inglês. Esse achado tem implicações para o design de sistemas RAG institucionais em língua portuguesa.

3. **A Faithfulness elevada e consistente (≥0.937) em todos os modos** indica que o sistema raramente alucina — as afirmações presentes nas respostas são sustentadas pelos documentos recuperados, independentemente da estratégia de recuperação utilizada. Isso confirma que o Gemini 2.5 Flash, com o prompt de síntese adotado, é eficaz em manter fidelidade ao contexto.

4. **A avaliação qualitativa confirma os resultados quantitativos:** o Context Recall alto do modo hybrid se traduz em respostas perceptivelmente mais completas em testes reais, e as melhorias de humanização (detecção de saudações, buffer de histórico, interceptação de fallbacks em inglês) produziram comportamento adequado em interações fora do escopo da recuperação.

5. **As limitações do conjunto de avaliação (20 questões, construção manual por único anotador)** impõem cautela na interpretação dos valores absolutos. Os resultados são mais confiáveis como indicadores da *direção* das diferenças entre modos do que como medidas absolutas de desempenho do sistema.
