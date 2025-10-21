# MLP para Classificação de Jogadores NBA

Uma aplicação de rede neural MLP (Multi-Layer Perceptron) implementada do zero em Python para classificar jogadores da NBA como "Good" ou "Bad" baseado em suas estatísticas de desempenho.

## 🏀 Sobre o Projeto

Este projeto utiliza uma implementação própria de MLP (sem bibliotecas de machine learning) para analisar dados de jogadores da NBA 2024 e classificá-los automaticamente baseado em 26 estatísticas diferentes.

## 📊 Resultados Obtidos

### Arquitetura da Rede Neural:
- **4 camadas** total: [26, 16, 8, 1]
- **Camada de Entrada**: 26 neurônios (26 estatísticas NBA)
- **Camadas Ocultas**: 2 camadas com 16 e 8 neurônios
- **Camada de Saída**: 1 neurônio (classificação binária)

### Detalhamento dos Parâmetros (Total: 577):
#### **Pesos (Weights):**
- **Camada 1→2**: 26×16 = 416 pesos (entrada para 1ª oculta)
- **Camada 2→3**: 16×8 = 128 pesos (1ª oculta para 2ª oculta)  
- **Camada 3→4**: 8×1 = 8 pesos (2ª oculta para saída)
- **Total de Pesos**: 552

#### **Bias (Viés):**
- **1ª Camada Oculta**: 16 bias (um por neurônio)
- **2ª Camada Oculta**: 8 bias (um por neurônio)
- **Camada de Saída**: 1 bias (um por neurônio)
- **Total de Bias**: 25

### Performance:
- **Acurácia de Teste**: 92.7%
- **Acurácia de Treino**: 94.8%
- **Redução do Loss**: 86.5%
- **Dataset**: 624 jogadores NBA

## 📁 Arquivos

- `mlp.py`: Implementação da classe MLP do zero
- `nba_mlp.py`: Aplicação principal para classificação NBA
- `nba_dados_2024.csv`: Dataset com estatísticas dos jogadores NBA 2024
- `nba_mlp_resultados.png`: Visualizações dos resultados
- `README.md`: Este arquivo

## 🚀 Como Executar

### Pré-requisitos
```bash
# Ativar ambiente virtual (se disponível)
source venv/bin/activate

# Ou usar Python diretamente
python3 nba_mlp.py
```

### Execução
```bash
./venv/bin/python nba_mlp.py
```

## 📈 Visualizações Geradas

O programa gera automaticamente o arquivo `nba_mlp_resultados.png` contendo:

1. **Curva de Loss**: Evolução do erro durante treinamento
2. **Distribuições**: Comparação entre dados reais e predições
3. **Matrizes de Confusão**: Para conjuntos de treino e teste
4. **Comparação de Acurácias**: Treino vs Teste

## 🧠 Características Técnicas da MLP

### Implementação Própria:
- **Sem bibliotecas de ML**: Apenas NumPy para operações matemáticas
- **Forward Pass**: Propagação direta através da rede
- **Backpropagation**: Algoritmo de retropropagação completo

### Funções Matemáticas:
#### **Funções de Ativação:**
- **ReLU** (camadas ocultas): f(x) = max(0, x)
  - Derivada: f'(x) = 1 se x > 0, senão 0
- **Sigmoid** (saída): f(x) = 1/(1 + e^(-x))
  - Derivada: f'(x) = f(x) × (1 - f(x))

#### **Função de Loss:**
- **Binary Cross-Entropy**: L = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
- **Gradiente**: ∂L/∂ŷ = ŷ - y

#### **Inicialização de Pesos:**
- **Xavier/Glorot**: W ~ N(0, √(2/n_entrada))
- **Objetivo**: Manter variância constante entre camadas
- **Bias**: Inicializados com zeros

#### **Otimização:**
- **Algoritmo**: Gradient Descent
- **Update Rule**: W = W - α × ∇W
- **Taxa de Aprendizado (α)**: 0.01

### Parâmetros Configuráveis:
- `layers`: Arquitetura da rede (ex: [26, 16, 8, 1])
- `learning_rate`: Taxa de aprendizado (0.01)
- `epochs`: Número de épocas de treinamento (1500)

## 📊 Dataset NBA

### Estatísticas Utilizadas (26 features):
- **Básicas**: Age, G, GS, MP, PTS
- **Arremessos**: FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%
- **Lance Livre**: FT, FTA, FT%
- **Rebotes**: ORB, DRB, TRB
- **Outras**: AST, STL, BLK, TOV, PF, eFG%

### Distribuição das Classes:
- **Good**: 52 jogadores (8.3%)
- **Bad**: 570 jogadores (91.7%)
- **Total**: 624 jogadores

## 🔧 Métodos Principais da MLP

```python
# Criar MLP
mlp = MLP(layers=[26, 16, 8, 1], learning_rate=0.01)

# Treinar
losses = mlp.train(X_train, y_train, epochs=1500)

# Predições
predictions = mlp.predict(X_test)          # Binário (0/1)
probabilities = mlp.predict_proba(X_test)  # Probabilidades (0-1)
```

## 📋 Pré-processamento dos Dados

1. **Limpeza**: Tratamento de valores ausentes e inválidos
2. **Normalização**: Z-score para todas as features numéricas
3. **Codificação**: Performance → Good (1) / Bad (0)
4. **Divisão**: 80% treino / 20% teste (estratificada)

## 🎯 Aplicações Futuras

- **Scouting**: Identificação automática de talentos
- **Análise de Desempenho**: Avaliação objetiva de jogadores
- **Mercado**: Suporte para decisões de contratação
- **Desenvolvimento**: Identificação de áreas de melhoria

## 🔍 Limitações Atuais

- Classificação binária apenas (Good/Bad)
- Dataset de uma única temporada (2024)
- Sem consideração de contexto (time, posição específica)
- Otimizador simples (gradient descent)

## 🚀 Melhorias Possíveis

- **Classificação multiclasse** (Excelente/Bom/Regular/Ruim)
- **Otimizadores avançados** (Adam, RMSprop)
- **Regularização** (L1, L2, Dropout)
- **Validação cruzada** e early stopping
- **Features engineered** (eficiência, clutch performance)
- **Análise temporal** (múltiplas temporadas)

## 📊 Dependências

- **NumPy**: Operações matemáticas e arrays
- **Matplotlib**: Visualizações e gráficos
- **Python 3.x**: Linguagem base

---

**Desenvolvido com MLP implementada do zero - Sem frameworks de ML! 🧠⚡**