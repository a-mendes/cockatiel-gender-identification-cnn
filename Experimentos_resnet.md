# Experimentos com ResNet no problema de identificação de gênero em calopsitas

Este documento resume os experimentos 0 e 1 do notebook `resnet-vs-efficientnet-cockatiel-gender-identification.ipynb`, destacando arquitetura, hiperparâmetros, estratégia de treinamento e resultados observados.

## Visão geral

Os dois experimentos usam a ResNet50 como backbone pré-treinado em ImageNet e partem do mesmo objetivo: classificar imagens de calopsitas em duas classes, `female` e `male`. A principal diferença não está apenas no modelo, mas na forma como o treinamento é organizado.

O experimento 0 é um pipeline mais direto: treina um classificador binário com uma fase inicial com a base congelada e depois faz fine-tuning parcial em um conjunto fixo de treino, validação e teste.

O experimento 1 mantém a ResNet50, mas muda a estratégia de aprendizado: usa validação cruzada estratificada, uma arquitetura multiescala e treinamento em duas fases por fold. Na prática, ele explora melhor o conjunto disponível e reduz a dependência de uma única divisão treino/validação.

## Experimento 0: ResNet50 base + fine-tuning

### Arquitetura

- Backbone: ResNet50 com pesos do ImageNet.
- `include_top=False`, para remover a cabeça original da ImageNet.
- Entrada: imagens redimensionadas para 224 x 224 x 3.
- Cabeça de classificação:
  - GlobalAveragePooling2D
  - Dropout com taxa 0.3
  - Dense com 32 neurônios e ativação ReLU
  - Dropout com taxa 0.2
  - Dense final com 1 neurônio e ativação sigmoid

### Hiperparâmetros e treinamento

- Pré-processamento: `ImageDataGenerator(rescale=1./255)`.
- Tamanho do batch: 64.
- Treinamento inicial:
  - backbone congelado
  - otimizador Adam com taxa de aprendizado 1e-3
  - loss binária: `binary_crossentropy`
  - 10 épocas
  - uso de `class_weight`
- Fine-tuning:
  - backbone descongelado parcialmente
  - apenas as últimas 30 camadas permanecem treináveis
  - otimizador Adam com taxa de aprendizado 1e-5
  - loss binária: `binary_crossentropy`
  - 10 épocas
  - uso de `class_weight`
- Limiar de decisão:
  - o notebook estima um limiar ótimo com base na curva precision-recall da validação.

### Resultado observado

Na matriz de confusão do teste, o modelo apresentou forte tendência a prever `male`:

- True `female` classificadas como `female`: 0
- True `female` classificadas como `male`: 10
- True `male` classificadas como `female`: 1
- True `male` classificadas como `male`: 12

Isso corresponde a uma acurácia aproximada de 52,2% no teste. O comportamento indica viés para a classe `male`, com alta sensibilidade para essa classe, mas baixa capacidade de reconhecer `female`.

## Experimento 1: ResNet50 com validação cruzada e arquitetura multiescala

### Arquitetura

- Backbone: ResNet50 com pesos do ImageNet.
- `include_top=False`.
- Entrada: 224 x 224 x 3.
- Arquitetura multiescala:
  - extração de features intermediárias em `conv4_block6_out`
  - GlobalAveragePooling2D sobre essas features intermediárias
  - GlobalAveragePooling2D sobre a saída final da rede
  - concatenação das duas representações
  - Dense com 128 neurônios e ativação ReLU
  - Dropout com taxa 0.4
  - Dense final com 2 neurônios e ativação softmax

### Hiperparâmetros e treinamento

- Estratégia de dados:
  - o pool de cross validation junta treino e validação segmentados
  - o teste permanece separado
- Validação cruzada:
  - StratifiedKFold com 5 folds
  - batch size 32
  - class mode `sparse`
- Data augmentation online nos folds de treino:
  - horizontal flip
  - rotation_range = 10
  - zoom_range = 0.1
  - width_shift_range = 0.05
  - height_shift_range = 0.05
  - brightness_range = (0.9, 1.1)
- Fase 1 de cada fold:
  - backbone congelado
  - Adam com learning rate 1e-3
  - loss: `sparse_categorical_crossentropy`
  - até 15 épocas
  - EarlyStopping com paciência 4
  - ReduceLROnPlateau com fator 0.2
- Fase 2 de cada fold:
  - backbone parcialmente descongelado
  - as primeiras 100 camadas continuam congeladas
  - Adam com learning rate 5e-6
  - loss: `sparse_categorical_crossentropy`
  - até 30 épocas
  - EarlyStopping com paciência 4
  - ReduceLROnPlateau com fator 0.2

### Resultado observado

A validação cruzada mostrou desempenho bem mais estável do que o experimento 0. Pela matriz out-of-fold:

- True `female` classificadas como `female`: 45
- True `female` classificadas como `male`: 17
- True `male` classificadas como `female`: 16
- True `male` classificadas como `male`: 58

Isso resulta em acurácia aproximada de 75,7% no conjunto out-of-fold. Na avaliação final no teste, a matriz de confusão foi:

- True `female` classificadas como `female`: 7
- True `female` classificadas como `male`: 3
- True `male` classificadas como `female`: 2
- True `male` classificadas como `male`: 11

A acurácia de teste fica em torno de 78,3%. Em termos práticos, o modelo acertou bem melhor as duas classes e reduziu a assimetria observada no experimento 0.

## Comparação direta

### Semelhanças

- Ambos usam ResNet50 pré-treinada em ImageNet.
- Ambos removem a cabeça original da rede e treinam uma nova cabeça de classificação.
- Ambos aplicam fine-tuning após uma fase inicial com o backbone congelado.
- Ambos usam data augmentation para aumentar a robustez do modelo.
- Ambos avaliam o desempenho com matriz de confusão e métricas de classificação.

### Diferenças principais

- O experimento 0 usa uma única divisão treino/validação/teste; o experimento 1 usa validação cruzada estratificada.
- O experimento 0 trabalha com uma cabeça binária simples; o experimento 1 usa uma cabeça multiescala com concatenação de features intermediárias e finais.
- O experimento 0 usa saída sigmoid com loss binária; o experimento 1 usa saída softmax com loss categórica esparsa.
- O experimento 0 depende de um limiar de decisão para converter probabilidades em classes; o experimento 1 decide diretamente por argmax.
- O experimento 1 explora melhor os dados disponíveis, porque treina sobre vários folds e consolida o comportamento médio do modelo.

## Hipóteses para a superioridade do experimento 1

O ganho de acurácia do experimento 1 em relação ao experimento 0 pode ser explicado por uma combinação de fatores.

1. A validação cruzada estratificada reduz a dependência de uma única partição dos dados. Em problemas com poucos exemplos, uma divisão específica pode favorecer ou prejudicar fortemente o resultado final.

2. O experimento 1 usa um pool maior de imagens de treino e validação para compor a cross validation, o que tende a melhorar a generalização.

3. A arquitetura multiescala ajuda a capturar padrões em diferentes níveis de abstração. Isso é útil quando as diferenças visuais entre as classes são sutis.

4. A combinação de EarlyStopping e ReduceLROnPlateau em cada fold torna o treinamento mais estável e menos sujeito a overfitting precoce.

5. A decisão multiclasse com softmax pode ter sido mais estável do que o esquema binário com sigmoid e limiar ótimo no experimento 0, sobretudo diante de poucas amostras de teste.

6. O experimento 0 mostra um viés forte para a classe `male`, sugerindo que a cabeça simples não capturou adequadamente as pistas visuais da classe `female`. O experimento 1 equilibra melhor esse comportamento.

## Conclusão

Os dois experimentos confirmam que a ResNet50 é uma base viável para o problema, mas o desempenho depende fortemente da estratégia de treinamento. O experimento 0 funciona como baseline, porém ficou mais instável e mais enviesado para uma classe. Já o experimento 1, ao combinar cross validation, arquitetura multiescala e fine-tuning mais controlado, entregou um resultado superior e mais confiável.

Se o objetivo for escolher a melhor abordagem dentro deste notebook, o experimento 1 é a opção mais consistente.
