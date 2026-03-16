# Transformer Completo em NumPy --- Laboratório 4

Este repositório contém uma implementação educacional de um
**Transformer completo (Encoder + Decoder)** utilizando **Python e
NumPy**.

O projeto foi desenvolvido como parte do **Laboratório 4 da disciplina
Tópicos em Inteligência Artificial**, ministrada no **ICEV**.

O objetivo do laboratório é demonstrar o funcionamento interno da
arquitetura **Transformer**, conforme apresentado no artigo:

Vaswani et al., 2017 --- *Attention Is All You Need*

A implementação reproduz os principais componentes da arquitetura:

-   Encoder Stack
-   Decoder Stack
-   Self-Attention
-   Masked Self-Attention
-   Cross-Attention
-   Feed Forward Network
-   Layer Normalization
-   Loop auto-regressivo de inferência

O foco do projeto não é treinamento do modelo, mas sim **entender a
estrutura e a mecânica matemática do Transformer**.

------------------------------------------------------------------------

# Estrutura do Projeto

    transformer-completo
    │
    ├── data.py
    ├── attention.py
    ├── encoder.py
    ├── decoder.py
    ├── transformer.py
    └── README.md

Cada arquivo implementa uma parte específica do modelo.

  Arquivo          Função
  ---------------- -------------------------------------
  data.py          Preparação dos dados e embeddings
  attention.py     Implementação matemática da atenção
  encoder.py       Camadas do Encoder
  decoder.py       Camadas do Decoder
  transformer.py   Integração completa do modelo

------------------------------------------------------------------------

# Preparação dos Dados

Arquivo: **data.py**

Este módulo realiza a preparação inicial da frase utilizada no
laboratório.

Etapas executadas:

-   definição do vocabulário
-   tokenização
-   conversão palavra → ID
-   criação da matriz de embeddings
-   geração do tensor de entrada `X`

Frase utilizada:

    No i am your father .

Cada palavra é convertida em um **ID numérico** e posteriormente em um
**vetor de embedding**.

Saída gerada pelo script:

    Vocabulary → token → id
    Embedding matrix shape → (6, 64)
    X shape → (1, 6, 64)

Significado das dimensões:

    batch_size = 1
    sequence_length = 6
    embedding_dimension = 64

------------------------------------------------------------------------

# Motor Matemático do Transformer

Arquivo: **attention.py**

Este módulo implementa os principais componentes matemáticos utilizados
pelo Encoder e Decoder.

## Softmax

Utilizada para converter logits em probabilidades.

    softmax(z)

------------------------------------------------------------------------

## Scaled Dot-Product Attention

Fórmula central do Transformer:

    Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

Função implementada:

    scaled_dot_product_attention(Q, K, V)

------------------------------------------------------------------------

## Self Attention

Geração das matrizes:

    Q = XWQ
    K = XWK
    V = XWV

Aplicação do mecanismo de atenção.

Função:

    self_attention(X)

------------------------------------------------------------------------

## Feed Forward Network

Rede neural aplicada posição por posição:

    FFN(x) = max(0, xW1 + b1)W2 + b2

Função:

    feed_forward(X)

------------------------------------------------------------------------

## Layer Normalization

Normaliza os valores para estabilizar o fluxo numérico do modelo.

    layer_norm(X)

------------------------------------------------------------------------

## Add & Norm

Implementa as conexões residuais do Transformer.

    Add & Norm = LayerNorm(X + Sublayer(X))

Função:

    add_and_norm(X, sublayer_output)

------------------------------------------------------------------------

# Encoder

Arquivo: **encoder.py**

Este módulo implementa a pilha do Encoder.

Cada camada executa:

    Self Attention
    ↓
    Add & Norm
    ↓
    Feed Forward Network
    ↓
    Add & Norm

No laboratório são utilizadas **6 camadas de Encoder**.

Saída do Encoder:

    Z

Este tensor contém as **representações contextualizadas da frase de
entrada**.

Exemplo de validação exibida:

    Dimensões de X mantidas: (1, 6, 64)
    Representações contextualizadas geradas: vetor Z obtido após o processamento pelo Encoder
    VALIDAÇÃO DE SANIDADE: PASSOU EM TODAS AS VERIFICAÇÕES

------------------------------------------------------------------------

# Decoder

Arquivo: **decoder.py**

O Decoder possui três componentes principais.

## 1. Masked Self Attention

Utiliza uma máscara causal que impede o modelo de acessar tokens
futuros.

Exemplo da máscara:

    [[ 0 -inf -inf -inf]
     [ 0 0 -inf -inf]
     [ 0 0 0 -inf]
     [ 0 0 0 0 ]]

------------------------------------------------------------------------

## 2. Cross Attention

Permite que o Decoder consulte a saída do Encoder.

    Q → Decoder
    K,V → Encoder

------------------------------------------------------------------------

## 3. Feed Forward Network

Mesmo mecanismo utilizado no Encoder.

------------------------------------------------------------------------

# Loop Auto-Regressivo

O Decoder gera a frase palavra por palavra.

    <START>
    ↓
    Predição da próxima palavra
    ↓
    Concatenação na sequência
    ↓
    Repetição do processo

O processo termina quando o token:

    <EOS>

é gerado.

------------------------------------------------------------------------

# Integração do Modelo Completo

Arquivo: **transformer.py**

Este script integra todos os componentes do Transformer.

Fluxo completo:

    Encoder Input
    ↓
    Encoder Stack
    ↓
    Representação contextual Z
    ↓
    Decoder
    ↓
    Softmax
    ↓
    Nova palavra
    ↓
    Concatenação na sequência
    ↓
    Loop até <EOS>

Entrada simulada utilizada no laboratório:

    Thinking Machines

Exemplo de saída:

    <START> Thinking father am <START> father father No . am No your <EOS>

Como os pesos são aleatórios (modelo não treinado), a sequência gerada
não possui significado semântico real.

O objetivo do laboratório é validar o fluxo estrutural do Transformer.

------------------------------------------------------------------------

# Como Executar

1.  Ativar o ambiente virtual:

```{=html}
<!-- -->
```
    venv\Scripts\activate

2.  Executar os módulos:

```{=html}
<!-- -->
```
    python data.py
    python attention.py
    python encoder.py
    python decoder.py
    python transformer.py

------------------------------------------------------------------------

# Tecnologias Utilizadas

-   Python
-   NumPy
-   Pandas

------------------------------------------------------------------------

# Observação sobre Uso de Inteligência Artificial

Partes do código foram auxiliadas por ferramentas de Inteligência
Artificial para revisão, curadoria e adequação aos requisitos
estabelecidos no laboratório.

Todo o código foi revisado manualmente para garantir aderência às
especificações da atividade.

------------------------------------------------------------------------

# Autor

Carlos Eugênio\
Engenharia de Software --- ICEV

Disciplina: Tópicos em Inteligência Artificial
