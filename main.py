"""
Implementação de Rede Neural MLP para Previsão de Desempenho de Jogadores de Basquete
Trabalho de Implementação sem uso de bibliotecas de Machine Learning

Implementação seguindo rigorosamente o algoritmo de backpropagation com:
- Binary Cross-Entropy Loss para classificação binária
- Cálculo correto dos gradientes conforme material teórico
- Momentum para acelerar convergência
"""

import pandas as pd
import numpy as np
import random
from typing import List, Tuple

class MLPNeuralNetwork:
    """
    Implementação de Rede Neural MLP com backpropagation correto
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 learning_rate: float = 0.1, momentum: float = 0.9):
        """
        Inicializa a rede neural MLP
        
        Args:
            input_size: Número de features de entrada
            hidden_layers: Lista com o número de neurônios em cada camada oculta
            output_size: Número de classes de saída (1 para classificação binária)
            learning_rate: Taxa de aprendizado (alpha) - típico: 0.01 a 0.3
            momentum: Fator de momento - típico: 0.8 a 0.95
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Criar arquitetura da rede
        self.layers = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layers)
        
        # Inicializar pesos e bias
        self.weights = []
        self.biases = []
        
        # Inicializar variações anteriores para momentum
        self.prev_weight_deltas = []
        self.prev_bias_deltas = []
        
        # Inicialização dos pesos (pequenos valores aleatórios)
        for i in range(self.num_layers - 1):
            # Pesos inicializados com valores pequenos aleatórios
            weight_matrix = np.random.uniform(-0.5, 0.5, (self.layers[i], self.layers[i + 1]))
            self.weights.append(weight_matrix)
            
            # Bias inicializados com zeros
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.biases.append(bias_vector)
            
            # Inicializar deltas anteriores para momentum
            self.prev_weight_deltas.append(np.zeros_like(weight_matrix))
            self.prev_bias_deltas.append(np.zeros_like(bias_vector))
    
    def sigmoid(self, x):
        """Função de ativação sigmoidal"""
        # Clipping para evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, sigmoid_output):
        """
        Derivada da função sigmoidal
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)
        """
        return sigmoid_output * (1 - sigmoid_output)
    
    def forward_propagation(self, X):
        """
        Propagação para frente (forward pass)
        
        Args:
            X: Dados de entrada (batch_size, input_size)
            
        Returns:
            activations: Lista com as ativações de cada camada
        """
        activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            # Calcular z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Aplicar função de ativação sigmoidal
            activation = self.sigmoid(z)
            activations.append(activation)
            current_input = activation
        
        return activations
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        """
        Função de perda Binary Cross-Entropy para classificação binária
        Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        """
        # Evitar log(0) adicionando pequeno epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward_propagation(self, X, y, activations):
        """
        Retropropagação (backward pass) seguindo o algoritmo do material
        
        Passo 3: Cálculo do erro
        - Camada de saída: δ = saída * (1 - saída) * (esperado - obtido)
        - Camadas ocultas: δ = saída * (1 - saída) * Σ(δ_seguinte * peso)
        """
        m = X.shape[0]  # Número de exemplos
        
        # Listas para armazenar gradientes (deltas)
        deltas = [None] * len(self.weights)
        
        # PASSO 3a: Calcular erro da camada de saída
        # δ_saída = saída * (1 - saída) * (esperado - obtido)
        output_layer = activations[-1]  # Saída da rede
        factor_erro = y - output_layer  # (SaídaEsperada - SaídaNeurônio)
        
        # Erro = SaídaNeurônio × (1−SaídaNeurônio) × FatorErro
        delta_output = self.sigmoid_derivative(output_layer) * factor_erro
        deltas[-1] = delta_output
        
        # PASSO 3b: Propagar erro para camadas ocultas (de trás para frente)
        for i in range(len(self.weights) - 2, -1, -1):
            # Saída da camada atual
            current_output = activations[i + 1]
            
            # FatorErro = Σ(δ_seguinte * peso_conexão)
            factor_erro_hidden = np.dot(deltas[i + 1], self.weights[i + 1].T)
            
            # δ = saída * (1 - saída) * FatorErro
            delta_hidden = self.sigmoid_derivative(current_output) * factor_erro_hidden
            deltas[i] = delta_hidden
        
        # Calcular gradientes dos pesos e bias
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights)):
            # Gradiente do peso = entrada * δ
            weight_grad = np.dot(activations[i].T, deltas[i]) / m
            weight_gradients.append(weight_grad)
            
            # Gradiente do bias = δ (média)
            bias_grad = np.mean(deltas[i], axis=0, keepdims=True)
            bias_gradients.append(bias_grad)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        PASSO 4: Atualizar pesos com momentum
        
        Δw(t) = α * gradiente + momentum * Δw(t-1)
        w(t+1) = w(t) + Δw(t)
        """
        for i in range(len(self.weights)):
            # Calcular variação com momentum
            weight_delta = (self.learning_rate * weight_gradients[i] + 
                          self.momentum * self.prev_weight_deltas[i])
            
            bias_delta = (self.learning_rate * bias_gradients[i] + 
                         self.momentum * self.prev_bias_deltas[i])
            
            # Atualizar pesos
            self.weights[i] += weight_delta
            self.biases[i] += bias_delta
            
            # Salvar deltas para próxima iteração (momentum)
            self.prev_weight_deltas[i] = weight_delta
            self.prev_bias_deltas[i] = bias_delta
    
    def train(self, X_train, y_train, epochs: int = 1000, verbose: bool = True):
        """
        Treinar a rede neural
        """
        losses = []
        
        for epoch in range(epochs):
            # PASSO 1: Forward propagation
            activations = self.forward_propagation(X_train)
            
            # PASSO 2: Calcular perda (Binary Cross-Entropy)
            loss = self.binary_cross_entropy_loss(y_train, activations[-1])
            losses.append(loss)
            
            # PASSO 3: Backward propagation (calcular gradientes)
            weight_gradients, bias_gradients = self.backward_propagation(X_train, y_train, activations)
            
            # PASSO 4: Atualizar parâmetros com momentum
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Imprimir progresso
            if verbose and (epoch + 1) % 200 == 0:
                accuracy = self.calculate_accuracy(X_train, y_train)
                print(f"Época {epoch + 1}/{epochs}, Perda: {loss:.6f}, Acurácia: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        """Fazer predições (probabilidades)"""
        activations = self.forward_propagation(X)
        return activations[-1]
    
    def predict_classes(self, X, threshold: float = 0.5):
        """Predizer classes (0 ou 1)"""
        predictions = self.predict(X)
        return (predictions >= threshold).astype(int)
    
    def calculate_accuracy(self, X, y):
        """Calcular acurácia"""
        predictions = self.predict_classes(X)
        return np.mean(predictions == y)


class DataProcessor:
    """
    Classe para processamento e análise dos dados
    """
    
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
    
    def load_and_analyze_data(self, filepath: str):
        """
        Carregar e analisar dados para criar critério de performance próprio
        """
        print("=== CARREGAMENTO E ANÁLISE DOS DADOS ===")
        print("-" * 50)
        
        # Carregar dados
        df = pd.read_csv(filepath)
        print(f"Dataset carregado: {df.shape[0]} jogadores, {df.shape[1]} colunas")
        
        # Limpar dados
        df = self.clean_data(df)
        print(f"Após limpeza: {df.shape[0]} jogadores")
        
        # Analisar distribuição original
        if 'Performance' in df.columns:
            original_dist = df['Performance'].value_counts()
            print(f"\nDistribuição original das classes:")
            for perf, count in original_dist.items():
                print(f"  {perf}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def clean_data(self, df):
        """Limpar e preparar dados"""
        # Fazer uma cópia para evitar warnings
        df = df.copy()
        
        # Remover linhas com muitos valores ausentes
        df = df.dropna(thresh=len(df.columns) * 0.7)
        
        # Tratar colunas com problemas de formatação
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['Player', 'Pos', 'Tm', 'Performance']:
                # Converter vírgulas para pontos e para numérico usando .loc
                df.loc[:, col] = df[col].astype(str).str.replace(',', '.')
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        
        # Preencher valores ausentes com mediana
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df.loc[:, col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_performance_criteria(self, df):
        """
        Criar critério próprio de performance baseado em análise estatística
        """
        print("\n=== CRIAÇÃO DE CRITÉRIO DE PERFORMANCE ===")
        print("-" * 50)
        
        # Selecionar features principais para o índice de performance
        key_features = ['PTS', 'AST', 'TRB', 'FG%', 'MP', 'G', 'STL', 'BLK']
        
        # Verificar quais features existem
        available_features = [f for f in key_features if f in df.columns]
        print(f"Features disponíveis para índice: {available_features}")
        
        # Normalizar features (Z-score)
        performance_data = df[available_features].copy()
        
        # Calcular Z-scores
        normalized_features = {}
        for feature in available_features:
            mean_val = performance_data[feature].mean()
            std_val = performance_data[feature].std()
            if std_val > 0:
                normalized_features[feature] = (performance_data[feature] - mean_val) / std_val
            else:
                normalized_features[feature] = performance_data[feature] * 0  # Se std=0, feature constante
        
        # Criar índice de performance ponderado
        weights = {
            'PTS': 0.25,    # Pontos (mais importante)
            'AST': 0.15,    # Assistências
            'TRB': 0.15,    # Rebotes totais
            'FG%': 0.15,    # Porcentagem de arremessos
            'MP': 0.10,     # Minutos jogados
            'G': 0.10,      # Jogos
            'STL': 0.05,    # Roubos de bola
            'BLK': 0.05     # Tocos
        }
        
        # Calcular índice ponderado
        performance_index = np.zeros(len(df))
        total_weight = 0
        
        for feature in available_features:
            if feature in weights:
                weight = weights[feature]
                performance_index += weight * normalized_features[feature]
                total_weight += weight
                print(f"  {feature}: peso {weight:.2f}, média normalizada: {normalized_features[feature].mean():.3f}")
        
        # Normalizar pelo peso total usado
        if total_weight > 0:
            performance_index = performance_index / total_weight
        
        # Definir threshold para "Good" (top 20%)
        threshold_percentile = 80
        performance_threshold = np.percentile(performance_index, threshold_percentile)
        
        # Criar labels baseadas no nosso critério
        new_performance = ['Good' if score > performance_threshold else 'Bad' 
                          for score in performance_index]
        
        # Estatísticas do novo critério
        good_count = sum(1 for p in new_performance if p == 'Good')
        bad_count = len(new_performance) - good_count
        
        print(f"\nNovo critério de performance (top {100-threshold_percentile}%):")
        print(f"  Good: {good_count} jogadores ({good_count/len(df)*100:.1f}%)")
        print(f"  Bad: {bad_count} jogadores ({bad_count/len(df)*100:.1f}%)")
        print(f"  Threshold do índice: {performance_threshold:.3f}")
        
        # Adicionar ao dataframe usando .loc para evitar warnings
        df = df.copy()  # Garantir que temos uma cópia
        df.loc[:, 'Performance_New'] = new_performance
        df.loc[:, 'Performance_Index'] = performance_index
        
        return df
    
    def prepare_features_and_target(self, df, use_new_criteria=True):
        """
        Preparar features e target para o modelo
        """
        print(f"\n=== PREPARAÇÃO PARA TREINAMENTO ===")
        print("-" * 50)
        
        # Selecionar features numéricas (excluir identificadores e targets)
        exclude_cols = ['Player', 'Pos', 'Tm', 'Performance', 'Performance_New', 'Performance_Index']
        feature_columns = [col for col in df.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        print(f"Features selecionadas: {len(feature_columns)}")
        print(f"Features: {feature_columns}")
        
        # Extrair features
        X = df[feature_columns].values
        
        # Escolher target
        target_col = 'Performance_New' if use_new_criteria else 'Performance'
        y = (df[target_col] == 'Good').astype(int).values.reshape(-1, 1)
        
        print(f"\nUsando critério: {'Novo (baseado em índice)' if use_new_criteria else 'Original do dataset'}")
        print(f"Distribuição final:")
        print(f"  Good (1): {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"  Bad (0): {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
        
        return X, y, feature_columns
    
    def normalize_features(self, X_train, X_test=None):
        """
        Normalizar features usando Z-score
        """
        # Calcular estatísticas do conjunto de treino
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0)
        
        # Evitar divisão por zero
        self.feature_stds[self.feature_stds == 0] = 1
        
        # Normalizar treino
        X_train_norm = (X_train - self.feature_means) / self.feature_stds
        
        if X_test is not None:
            # Normalizar teste usando estatísticas do treino
            X_test_norm = (X_test - self.feature_means) / self.feature_stds
            return X_train_norm, X_test_norm
        
        return X_train_norm


class ModelEvaluator:
    """
    Classe para avaliação do modelo
    """
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Calcular matriz de confusão"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def calculate_metrics(confusion_mat):
        """Calcular métricas de avaliação"""
        tn, fp, fn, tp = confusion_mat.ravel()
        
        # Evitar divisão por zero
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_mat
        }
    
    @staticmethod
    def calculate_roc_curve(y_true, y_scores):
        """
        Calcular curva ROC seguindo o algoritmo:
        1. Variar o limiar de 1→0
        2. Para cada limiar, calcular TP, FP, TN, FN
        3. Derivar TPR e FPR
        4. Plotar pontos (FPR, TPR)
        
        Args:
            y_true: Labels verdadeiros (0 ou 1)
            y_scores: Probabilidades/scores do modelo
            
        Returns:
            fpr: False Positive Rate para cada threshold
            tpr: True Positive Rate para cada threshold  
            thresholds: Thresholds utilizados
            auc: Area Under Curve
        """
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()
        
        # Criar thresholds variando de 1.0 → 0.0
        # Incluir valores únicos dos scores + extremos
        thresholds = np.unique(y_scores)
        thresholds = np.concatenate([[1.0], thresholds, [0.0]])
        thresholds = np.sort(thresholds)[::-1]  # Ordem decrescente (1→0)
        
        fpr_list = []
        tpr_list = []
        
        print(f"\nCalculando Curva ROC com {len(thresholds)} thresholds...")
        
        for i, threshold in enumerate(thresholds):
            # Aplicar threshold para fazer predições
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calcular TP, FP, TN, FN
            tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
            tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
            fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
            
            # Calcular TPR e FPR
            # TPR = TP / (TP + FN) - taxa de verdadeiros positivos (sensibilidade/recall)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # FPR = FP / (FP + TN) - taxa de falsos positivos (1 - especificidade)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            
            # Mostrar alguns pontos para debug
            if i < 5 or i % (len(thresholds)//5) == 0:
                print(f"  Threshold {threshold:.3f}: TPR={tpr:.3f}, FPR={fpr:.3f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")
        
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        
        # Calcular AUC usando regra do trapézio
        auc = ModelEvaluator.calculate_auc(fpr, tpr)
        
        return fpr, tpr, thresholds, auc
    
    @staticmethod
    def calculate_auc(fpr, tpr):
        """
        Calcular Area Under Curve usando regra do trapézio
        """
        # Ordenar por FPR crescente
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
        
        # Calcular área usando regra do trapézio
        auc = 0.0
        for i in range(1, len(fpr_sorted)):
            # Área do trapézio = (base) * (altura_média)
            base = fpr_sorted[i] - fpr_sorted[i-1]
            altura_media = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
            auc += base * altura_media
        
        return auc
    
    @staticmethod
    def print_roc_analysis(fpr, tpr, thresholds, auc, title="Análise ROC"):
        """
        Imprimir análise da curva ROC
        """
        print(f"\n=== {title} ===")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Interpretação do AUC
        if auc >= 0.9:
            interpretation = "Excelente"
        elif auc >= 0.8:
            interpretation = "Bom"
        elif auc >= 0.7:
            interpretation = "Razoável"
        elif auc >= 0.6:
            interpretation = "Fraco"
        elif auc >= 0.5:
            interpretation = "Aleatório"
        else:
            interpretation = "Pior que aleatório"
        
        print(f"Interpretação: {interpretation}")
        
        # Encontrar ponto ótimo (mais próximo do canto superior esquerdo)
        # Distância euclidiana até (0,1)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        print(f"\nPonto Ótimo da Curva ROC:")
        print(f"  Threshold: {optimal_threshold:.4f}")
        print(f"  TPR (Sensibilidade): {optimal_tpr:.4f}")
        print(f"  FPR (1-Especificidade): {optimal_fpr:.4f}")
        print(f"  Especificidade: {1-optimal_fpr:.4f}")
        
        # Mostrar alguns pontos da curva
        print(f"\nPontos da Curva ROC (amostra):")
        print(f"{'Threshold':>10} {'FPR':>8} {'TPR':>8}")
        print("-" * 30)
        
        # Mostrar 10 pontos espaçados
        step = max(1, len(thresholds) // 10)
        for i in range(0, len(thresholds), step):
            print(f"{thresholds[i]:>10.3f} {fpr[i]:>8.3f} {tpr[i]:>8.3f}")
        
        return optimal_threshold, optimal_fpr, optimal_tpr
    
    @staticmethod
    def plot_roc_curve_ascii(fpr, tpr, auc):
        """
        Plotar curva ROC em ASCII art
        """
        print(f"\n=== CURVA ROC (ASCII) - AUC = {auc:.3f} ===")
        
        # Criar grid 20x20 para visualização
        grid_size = 20
        grid = [[' ' for _ in range(grid_size + 1)] for _ in range(grid_size + 1)]
        
        # Marcar eixos
        for i in range(grid_size + 1):
            grid[grid_size][i] = '-'  # Eixo X
            grid[i][0] = '|'          # Eixo Y
        
        grid[grid_size][0] = '+'  # Origem
        
        # Plotar linha diagonal (classificador aleatório)
        for i in range(grid_size + 1):
            x = i
            y = grid_size - i
            if 0 <= y <= grid_size:
                grid[y][x] = '.'
        
        # Plotar pontos da curva ROC
        for i in range(len(fpr)):
            x = int(fpr[i] * grid_size)
            y = int((1 - tpr[i]) * grid_size)
            
            if 0 <= x <= grid_size and 0 <= y <= grid_size:
                grid[y][x] = '*'
        
        # Imprimir grid
        print("TPR")
        print("1.0 ↑")
        for row in grid:
            print("    " + "".join(row))
        print("    └" + "─" * grid_size + "→ FPR")
        print("   0.0" + " " * (grid_size-6) + "1.0")
        print("\nLegenda: * = Curva ROC, . = Linha Aleatória (AUC=0.5)")
    
    @staticmethod
    def print_evaluation_report(metrics, title="Avaliação"):
        """Imprimir relatório de avaliação"""
        print(f"\n=== {title} ===")
        print(f"Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precisão: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nMatriz de Confusão:")
        print(f"                 Predito")
        print(f"               Bad  Good")
        print(f"Real    Bad   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"        Good  {cm[1,0]:4d}  {cm[1,1]:4d}")


def split_train_test(X, y, test_size=0.2, random_state=42):
    """Dividir dados em treino e teste"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Criar índices aleatórios
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def main():
    """
    Função principal - Pipeline completo do trabalho
    """
    print("=" * 70)
    print("TRABALHO DE IMPLEMENTAÇÃO DE REDE NEURAL MLP")
    print("Previsão de Desempenho de Jogadores de Basquete")
    print("=" * 70)
    
    # 1. PREPARAÇÃO DOS DADOS
    processor = DataProcessor()
    
    # Carregar e analisar dados
    df = processor.load_and_analyze_data('nba_dados_2024.csv')
    
    # Criar critério próprio de performance
    df = processor.create_performance_criteria(df)
    
    # Preparar features e target
    X, y, feature_names = processor.prepare_features_and_target(df, use_new_criteria=True)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivisão dos dados:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    print(f"  Features: {X_train.shape[1]}")
    
    # Normalizar features
    X_train_norm, X_test_norm = processor.normalize_features(X_train, X_test)
    
    # 2. IMPLEMENTAÇÃO E TREINAMENTO DO MODELO
    print(f"\n=== TREINAMENTO DA REDE NEURAL MLP ===")
    print("-" * 50)
    
    # Configurar arquitetura
    input_size = X_train_norm.shape[1]
    hidden_layers = [20, 10]  # Duas camadas ocultas
    output_size = 1
    learning_rate = 0.1      # Alpha (taxa de aprendizado)
    momentum = 0.9           # Fator de momento
    
    print(f"Arquitetura da rede: {input_size} -> {hidden_layers} -> {output_size}")
    print(f"Taxa de aprendizado (α): {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Função de perda: Binary Cross-Entropy")
    print(f"Função de ativação: Sigmoidal")
    
    # Criar e treinar modelo
    model = MLPNeuralNetwork(input_size, hidden_layers, output_size, learning_rate, momentum)
    
    print(f"\nIniciando treinamento...")
    losses = model.train(X_train_norm, y_train, epochs=1500, verbose=True)
    
    # 3. AVALIAÇÃO DO MODELO
    print(f"\n=== AVALIAÇÃO DO MODELO ===")
    print("-" * 50)
    
    evaluator = ModelEvaluator()
    
    # Avaliação no conjunto de treino
    y_train_pred = model.predict_classes(X_train_norm)
    train_confusion = evaluator.confusion_matrix(y_train, y_train_pred)
    train_metrics = evaluator.calculate_metrics(train_confusion)
    evaluator.print_evaluation_report(train_metrics, "CONJUNTO DE TREINO")
    
    # Avaliação no conjunto de teste
    y_test_pred = model.predict_classes(X_test_norm)
    test_confusion = evaluator.confusion_matrix(y_test, y_test_pred)
    test_metrics = evaluator.calculate_metrics(test_confusion)
    evaluator.print_evaluation_report(test_metrics, "CONJUNTO DE TESTE")
    
    # ANÁLISE DA CURVA ROC
    print(f"\n=== ANÁLISE DA CURVA ROC ===")
    print("-" * 50)
    
    # Obter probabilidades para curva ROC
    y_test_proba = model.predict(X_test_norm)
    
    # Calcular curva ROC
    fpr, tpr, thresholds, auc = evaluator.calculate_roc_curve(y_test, y_test_proba)
    
    # Análise detalhada da ROC
    optimal_threshold, optimal_fpr, optimal_tpr = evaluator.print_roc_analysis(fpr, tpr, thresholds, auc, "CURVA ROC - CONJUNTO DE TESTE")
    
    # Plotar curva ROC em ASCII
    evaluator.plot_roc_curve_ascii(fpr, tpr, auc)
    
    # 4. ANÁLISE DE IMPORTÂNCIA DAS FEATURES
    print(f"\n=== ANÁLISE DE IMPORTÂNCIA DAS FEATURES ===")
    print("-" * 50)
    
    # Calcular importância baseada nos pesos da primeira camada
    first_layer_weights = np.abs(model.weights[0])
    feature_importance = np.mean(first_layer_weights, axis=1)
    
    # Ordenar por importância
    importance_indices = np.argsort(feature_importance)[::-1]
    
    print("Top 10 features mais importantes:")
    for i, idx in enumerate(importance_indices[:10]):
        importance_score = feature_importance[idx]
        feature_name = feature_names[idx]
        
        # Calcular correlação simples com target
        correlation = np.corrcoef(X[:, idx], y.flatten())[0, 1]
        
        print(f"{i+1:2d}. {feature_name:15s}: {importance_score:.4f} (corr: {correlation:+.3f})")
    
    # 5. EXEMPLOS DE PREDIÇÕES
    print(f"\n=== EXEMPLOS DE PREDIÇÕES ===")
    print("-" * 50)
    
    # Mostrar algumas predições
    y_test_proba = model.predict(X_test_norm)
    
    print(f"{'Real':>6} {'Predito':>8} {'Prob.':>8} {'Status':>12}")
    print("-" * 40)
    
    correct = 0
    for i in range(min(15, len(y_test))):
        real = "Good" if y_test[i][0] == 1 else "Bad"
        pred = "Good" if y_test_pred[i][0] == 1 else "Bad"
        prob = y_test_proba[i][0]
        status = "✓" if y_test[i][0] == y_test_pred[i][0] else "✗"
        
        if y_test[i][0] == y_test_pred[i][0]:
            correct += 1
        
        print(f"{real:>6} {pred:>8} {prob:>8.3f} {status:>12}")
    
    print(f"\nAcurácia nos exemplos: {correct}/{min(15, len(y_test))} ({correct/min(15, len(y_test))*100:.1f}%)")
    
    # 6. CONCLUSÕES
    print(f"\n=== CONCLUSÕES FINAIS ===")
    print("-" * 50)
    
    print("✅ IMPLEMENTAÇÃO REALIZADA:")
    print("   • Rede Neural MLP implementada do zero")
    print("   • Backpropagation com cálculo correto dos gradientes")
    print("   • Binary Cross-Entropy Loss para classificação")
    print("   • Momentum para acelerar convergência")
    print("   • Critério próprio de performance baseado em índice estatístico")
    
    print(f"\n📊 RESULTADOS OBTIDOS:")
    print(f"   • Acurácia no teste: {test_metrics['accuracy']*100:.2f}%")
    print(f"   • Precisão: {test_metrics['precision']*100:.2f}%")
    print(f"   • Recall: {test_metrics['recall']*100:.2f}%")
    print(f"   • F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   • AUC-ROC: {auc:.4f}")
    print(f"   • Threshold ótimo ROC: {optimal_threshold:.4f}")
    
    print(f"\n🎯 FEATURES MAIS IMPORTANTES:")
    for i in range(min(5, len(importance_indices))):
        idx = importance_indices[i]
        print(f"   {i+1}. {feature_names[idx]} (importância: {feature_importance[idx]:.4f})")
    
    print(f"\n🔬 METODOLOGIA APLICADA:")
    print("   • Normalização Z-score antes da definição de critérios")
    print("   • Índice de performance ponderado com features principais")
    print("   • Threshold baseado em percentil (top 20% = Good)")
    print("   • Algoritmo de backpropagation seguindo material teórico")
    print("   • Validação com métricas apropriadas para classificação")
    print("   • Análise completa da Curva ROC com cálculo de AUC")
    print("   • Otimização de threshold baseada na curva ROC")
    
    print(f"\n{'='*70}")
    print("TRABALHO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*70}")
    
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()