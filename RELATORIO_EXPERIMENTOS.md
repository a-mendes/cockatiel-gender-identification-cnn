# Relatório dos Experimentos

## Visão geral

Este notebook investiga a classificação de gênero em calopsitas com diferentes estratégias de aprendizado por transferência e ajustes progressivos de treinamento. O fluxo evoluiu ao longo das iterações para atacar os principais gargalos observados no problema:

- baixa generalização fora do conjunto de treino;
- tendência do classificador colapsar em uma única classe;
- sensibilidade excessiva ao fundo, objetos e composição visual das imagens;
- dificuldade em aproveitar bem um dataset pequeno.

A solução final do notebook combina três frentes principais:

- tratamento mais cuidadoso das imagens antes do treino;
- fine tuning em duas fases com backbones pré-treinados;
- avaliação mais rigorosa com threshold ajustado e, em seguida, cross validation estratificada.

## Organização do tratamento dos dados

### Download do dataset

O dataset foi baixado a partir de um arquivo ZIP hospedado no Google Drive, usando `gdown`. O notebook passou a trabalhar com a raiz local `cockatiel_gender_segmented`.

### Características do dataset

O dataset é composto por

Antes do augmentation, o notebook passou a gerar versões segmentadas das imagens para destacar a ave e reduzir a influência de elementos irrelevantes do cenário.

O pipeline de segmentação foi desenhado para ser mais tolerante nas bordas e mais robusto a falhas de máscara:

- tentativa inicial com saliency map para estimar regiões relevantes;
- fallback para `grabCut` quando a saliência não era suficiente;
- fechamento morfológico para reduzir buracos internos;
- flood fill para preencher regiões vazias;
- dilatação da máscara para preservar bordas da ave;
- crop com margem para não cortar cauda, asas e cabeça;
- padding mínimo para impedir que a imagem ficasse pequena demais para o augmentation.

Essa etapa foi importante porque o problema original mostrava forte sensibilidade ao fundo e a objetos externos. A intenção não foi remover tudo fora da ave de forma agressiva, mas garantir que a calopsita permanecesse sempre em evidência.

### Data augmentation

O augmentation foi aplicado depois da segmentação, sobre as imagens de treino já tratadas. As transformações usadas foram:

- rotação;
- zoom aleatório;
- espelhamento horizontal;
- escala de cinza;
- ajuste aleatório de cor.

Mais tarde, a rotação passou a usar `rotate_without_crop` com `resize` prévio, porque o método original de rotação do Augmentor gerava erros de crop em imagens mais recortadas. Essa troca estabilizou a geração de amostras.

## Experimento com ResNet50

### Objetivo

A primeira linha de experimento usou transfer learning com ResNet50 pré-treinada no ImageNet. A ideia era aproveitar uma backbone forte para extração de características visuais gerais e adaptar apenas a cabeça de classificação ao domínio das calopsitas.

### Arquitetura

A arquitetura final da ResNet50 ficou organizada assim:

- backbone `ResNet50(include_top=False, weights='imagenet')`;
- camada `GlobalAveragePooling2D`;
- camada densa intermediária com `32` neurônios e ativação `relu`;
- dropout `0.2`;
- camada de saída com `1` neurônio e ativação `sigmoid`.

Essa versão é a forma enxuta que ficou no notebook após a revisão de arquitetura. Antes disso, o experimento também passou por uma cabeça mais pesada, com duas camadas densas maiores. A análise posterior indicou que esse bloco denso maior tendia a aumentar a capacidade de memorização sem trazer ganho consistente de generalização.

### Estratégia de treino

O treino foi feito em duas fases:

1. **Warm-up**: backbone congelado, treinamento apenas da cabeça nova.
2. **Fine tuning**: descongelamento parcial das últimas camadas da ResNet50, com learning rate muito menor.

### Hiperparâmetros principais

- entrada: `224 x 224 x 3`;
- batch size: `64`;
- loss: `binary_crossentropy`;
- otimizador na fase inicial: `Adam(learning_rate=1e-3)`;
- otimizador no fine tuning: `Adam(learning_rate=1e-5)`;
- métricas: acurácia, precisão, recall e AUC;
- classes: `female` e `male`.

### Tratamento de avaliação

O notebook passou a calcular um threshold ótimo com base na validação, em vez de fixar `0.5`. O critério foi maximizar o F1-score na curva precisão-recall.

Isso foi importante porque o modelo tendia a colapsar em um limiar padrão que favorecia uma classe. O ajuste do threshold reduziu parte desse problema de decisão.

### Principais resultados observados

Nos testes do notebook, a ResNet50 mostrou:

- crescimento rápido da acurácia de treino, o que é esperado por usar uma backbone pré-treinada;
- forte dependência da qualidade da segmentação e do augmentation;
- tendência a concentrar previsões em uma classe quando a cabeça densa estava grande demais;
- melhora parcial ao reduzir a cabeça densa e ao usar threshold ajustado pela validação.

A conclusão prática foi que a ResNet50 aprendeu rapidamente padrões gerais, mas ainda sofria com generalização insuficiente quando a cabeça estava muito grande ou quando a imagem continha ruído visual relevante.

## Experimento com EfficientNet-B0

O notebook passou a conter dois usos da EfficientNet-B0:

- uma versão convencional com fine tuning em duas fases;
- uma versão mais elaborada, multiescala, para capturar padrões locais e globais ao mesmo tempo.

### 1. Versão convencional da EfficientNet-B0

#### Arquitetura

A EfficientNet-B0 foi usada com `include_top=False` e carregada com pesos do ImageNet. A cabeça final seguiu um padrão compacto:

- `GlobalAveragePooling2D`;
- `Dropout(0.3)`;
- `Dense(32, activation='relu')`;
- `Dropout(0.2)`;
- `Dense(1, activation='sigmoid')`.

#### Estratégia de treino

- fase 1: backbone congelado, learning rate `1e-3`, treino inicial;
- fase 2: fine tuning parcial, congelando as primeiras `100` camadas e usando learning rate `5e-6`;
- callbacks: `EarlyStopping` e `ReduceLROnPlateau`.

#### Pipeline de dados

- treino: imagens aumentadas e segmentadas;
- validação: imagens segmentadas;
- teste: imagens segmentadas;
- preprocessamento: `tf.keras.applications.efficientnet.preprocess_input`.

#### Resultados observados

No teste, essa versão atingiu aproximadamente:

- precisão: `0.5652`;
- recall: `1.0000`;
- F1-score: `0.7222`;
- threshold ótimo na validação: `0.1521`.

A matriz de confusão indicou um comportamento ainda enviesado para uma das classes, embora a eficiência do treino tenha melhorado em relação ao baseline anterior.

### 2. Versão multiescala da EfficientNet-B0

#### Motivação

Essa abordagem foi inspirada pela hipótese de que o gênero pode depender tanto de padrões finos, como textura e pequenos marcadores visuais, quanto de características mais globais, como postura e formato do corpo. Para isso, o modelo passou a combinar duas escalas de representação:

- uma representação intermediária, mais local;
- uma representação final, mais global.

#### Arquitetura

O modelo multiescala foi construído com:

- entrada `224 x 224 x 3`;
- augmentation inline com `RandomFlip`, `RandomRotation` e `RandomBrightness`;
- backbone `EfficientNetB0(include_top=False, weights='imagenet')`;
- extração da camada intermediária `block5c_add`;
- pooling global na saída intermediária;
- pooling global na saída final;
- concatenação das duas representações;
- camada densa com `128` neurônios;
- dropout `0.4`;
- saída com `2` neurônios e `softmax`.

#### Decisão arquitetural

Essa versão foi importante por três motivos:

- a saída em `softmax` simplifica o uso de `sparse_categorical_crossentropy`;
- a fusão de features locais e globais permite um modelo mais sensível a detalhes da ave;
- a cabeça final permanece relativamente enxuta, reduzindo a chance de overfitting.

#### Estratégia de treino

O treino foi dividido em duas fases:

1. **Treino inicial**: backbone congelado e head treinável.
2. **Fine tuning**: backbone parcialmente descongelado, mantendo congeladas as primeiras `100` camadas.

#### Hiperparâmetros principais

- input shape: `224 x 224 x 3`;
- batch size: `32`;
- loss: `sparse_categorical_crossentropy`;
- otimizador fase 1: `Adam(learning_rate=1e-3)`;
- otimizador fase 2: `Adam(learning_rate=5e-6)`;
- callbacks: `EarlyStopping` e `ReduceLROnPlateau`;
- augmentation no treino: flip horizontal, rotação, zoom, deslocamento e brilho.

#### Resultados observados

A EfficientNet multiescala apresentou um comportamento mais estável de treino do que as versões anteriores, mas ainda não resolveu completamente o viés do classificador. Em termos práticos:

- a acurácia de treino subiu de forma consistente;
- a validação estabilizou em valores moderados;
- o teste ainda mostrou tendência a favorecer uma classe;
- o F1 final continuou limitado pelo desbalanceamento decisório do classificador.

## Cross validation com EfficientNet-B0

### Objetivo

A cross validation foi adicionada para aproveitar também o conjunto de validação durante o treinamento, sem abandonar a avaliação final no conjunto de teste.

### Estrutura do protocolo

O protocolo implementado segue a lógica:

- juntar treino e validação segmentados em um pool único;
- dividir esse pool em `5` folds estratificados;
- treinar um modelo por fold;
- guardar as previsões out-of-fold;
- consolidar média e desvio padrão das métricas;
- por fim, treinar um modelo final com todo o pool e avaliar no teste separado.

### Modelo usado na cross validation

A cross validation reutiliza a EfficientNet-B0 multiescala, com:

- augmentation leve no treino de cada fold;
- pooling intermediário + pooling final;
- concatenação das duas escalas;
- cabeça compacta com `Dense(128)` e dropout;
- saída `softmax` com `2` classes.

### Métricas acompanhadas

Para cada fold, o notebook calcula:

- acurácia;
- precisão;
- recall;
- F1-score.

Depois, o notebook também agrega:

- média das métricas;
- desvio padrão;
- matriz de confusão out-of-fold.

### Decisões importantes

- `StratifiedKFold` foi escolhido para preservar a proporção das classes em cada fold.
- O teste ficou fora da cross validation para manter uma avaliação final realmente independente.
- O treino final usa o pool completo de treino+validação apenas após a análise dos folds.

## Comparação entre as abordagens

### ResNet50

Pontos fortes:

- backbone muito forte e conhecida;
- converge rápido;
- boa como baseline.

Limitações observadas:

- cabeça densa maior tendia a overfitting;
- sensível ao fundo e a ruídos visuais;
- generalização limitada sem ajuste cuidadoso do threshold e da segmentação.

### EfficientNet-B0

Pontos fortes:

- backbone mais eficiente em termos de parâmetros;
- adaptação mais elegante para datasets menores;
- melhor estabilidade com cabeça enxuta;
- versão multiescala amplia a capacidade de capturar padrões complementares.

Limitações observadas:

- ainda sensível à qualidade da segmentação;
- mesmo com fine tuning, o classificador pode manter viés para uma classe;
- melhorias reais dependem de um pipeline de dados bem controlado.

## Conclusões práticas

O notebook evoluiu de um experimento de transfer learning padrão para um pipeline mais elaborado, com foco em reduzir o ruído visual e tornar a avaliação mais confiável.

As principais lições foram:

- segmentar e padronizar melhor as imagens é decisivo;
- usar uma cabeça densa pequena tende a generalizar melhor;
- threshold fixo em `0.5` nem sempre é o melhor para um problema binário real;
- EfficientNet-B0 é uma boa candidata quando se quer eficiência e regularização;
- cross validation é útil para aproveitar melhor dados escassos e medir estabilidade.

## Resumo final dos parâmetros mais relevantes

### Pré-processamento

- resolução: `224 x 224`;
- normalização: `rescale=1./255` ou `preprocess_input` da EfficientNet;
- segmentação prévia com tolerância de borda;
- augmentation com rotação, zoom, flip, cor, brilho e deslocamentos leves.

### ResNet50

- backbone: `ResNet50`;
- cabeça: `GlobalAveragePooling2D -> Dense(32) -> Dropout(0.2) -> Dense(1)`;
- loss: `binary_crossentropy`;
- learning rates: `1e-3` e `1e-5`.

### EfficientNet-B0 convencional

- backbone: `EfficientNetB0`;
- cabeça: `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(32) -> Dropout(0.2) -> Dense(1)`;
- loss: `binary_crossentropy`;
- learning rates: `1e-3` e `5e-6`;
- threshold de decisão ajustado pela validação.

### EfficientNet-B0 multiescala

- backbone: `EfficientNetB0`;
- fusão de `block5c_add` com saída final;
- cabeça: `Dense(128) -> Dropout(0.4) -> Dense(2, softmax)`;
- loss: `sparse_categorical_crossentropy`;
- learning rates: `1e-3` e `5e-6`;
- fine tuning parcial das últimas camadas.

### Cross validation

- estratégia: `StratifiedKFold(n_splits=5)`;
- pooling de treino+validação segmentados;
- avaliação adicional com matriz out-of-fold;
- teste mantido separado para a avaliação final.

## Observação final

O notebook ficou mais próximo de um protocolo experimental completo do que de um simples treino de modelo. Ele já cobre:

- tratamento de dados;
- comparação de backbones;
- avaliação por threshold;
- cross validation;
- treino final com teste separado.

Isso fornece uma base mais séria para justificar as escolhas do trabalho e comparar as abordagens de forma metodologicamente consistente.
