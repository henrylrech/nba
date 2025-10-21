import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend para salvar arquivos
import matplotlib.pyplot as plt
from mlp import MLP

def load_and_preprocess_nba_data():
    """Carrega e preprocessa os dados da NBA usando apenas NumPy"""
    print("Carregando dados da NBA...")
    
    # Carregar dados usando NumPy
    data = []
    headers = []
    
    with open('nba_dados_2024.csv', 'r') as f:
        content = f.read()
        
        # Corrigir o problema da quebra de linha no cabeçalho
        content = content.replace('Performa\nnce', 'Performance')
        lines = content.strip().split('\n')
        
        # Primeira linha são os headers
        headers = lines[0].split(',')
        print(f"Colunas encontradas: {headers}")
        
        # Processar dados
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                continue
                
            # Parse manual para lidar com aspas
            row = []
            current_field = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    row.append(current_field.strip())
                    current_field = ""
                else:
                    current_field += char
            
            # Adicionar último campo
            row.append(current_field.strip())
            
            # Verificar se tem o número correto de colunas
            if len(row) == len(headers):
                data.append(row)
            elif i <= 5:  # Debug primeiras linhas
                print(f"Linha {i}: {len(row)} campos vs {len(headers)} esperados")
    
    if len(data) == 0:
        raise ValueError("Nenhum dado foi carregado!")
    
    data = np.array(data)
    print(f"Dataset carregado: {len(data)} jogadores, {len(headers)} colunas")
    
    # Encontrar índice da coluna Performance
    performance_idx = headers.index('Performance')
    performance_col = data[:, performance_idx]
    
    # Contar classes
    unique_classes, counts = np.unique(performance_col, return_counts=True)
    print(f"Classes de Performance: {unique_classes}")
    for cls, count in zip(unique_classes, counts):
        print(f"  {cls}: {count}")
    
    # Selecionar features numéricas (excluindo colunas de texto)
    numeric_indices = []
    feature_names = []
    
    # Colunas que sabemos que são numéricas
    numeric_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', 
                   '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
                   'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
                   'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    for col in numeric_cols:
        if col in headers:
            idx = headers.index(col)
            numeric_indices.append(idx)
            feature_names.append(col)
    
    print(f"Features numéricas selecionadas: {feature_names}")
    
    # Extrair features numéricas
    X_raw = data[:, numeric_indices]
    
    # Converter para float, tratando valores problemáticos
    X = np.zeros((X_raw.shape[0], X_raw.shape[1]))
    
    for i in range(X_raw.shape[0]):
        for j in range(X_raw.shape[1]):
            try:
                # Tratar valores com vírgula como separador decimal
                val = X_raw[i, j].replace(',', '.')
                X[i, j] = float(val)
            except:
                X[i, j] = 0.0  # Valor padrão para dados inválidos
    
    # Tratar valores NaN/infinitos
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Converter Performance para binário (Good=1, Bad=0)
    y = (performance_col == 'Good').astype(int).reshape(-1, 1)
    
    # Normalizar features (Z-score)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Evitar divisão por zero
    X_normalized = (X - X_mean) / X_std
    
    return X_normalized, y, feature_names, data

def split_data(X, y, test_size=0.2):
    """Divide dados em treino e teste"""
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Embaralhar índices
    indices = np.random.permutation(n_samples)
    
    # Dividir
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def plot_results(losses, y_train, y_pred_train, y_test, y_pred_test, feature_names):
    """Criar visualizações dos resultados"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Curva de Loss
    axes[0, 0].plot(losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Curva de Loss Durante Treinamento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuição de Predições - Treino
    axes[0, 1].hist([y_train[y_train==0].flatten(), y_train[y_train==1].flatten()], 
                    bins=2, alpha=0.7, label=['Bad (Real)', 'Good (Real)'], color=['red', 'green'])
    axes[0, 1].hist([y_pred_train[y_pred_train==0].flatten(), y_pred_train[y_pred_train==1].flatten()], 
                    bins=2, alpha=0.5, label=['Bad (Pred)', 'Good (Pred)'], color=['darkred', 'darkgreen'])
    axes[0, 1].set_title('Distribuição - Conjunto de Treino', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Classe')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].legend()
    
    # 3. Distribuição de Predições - Teste
    axes[0, 2].hist([y_test[y_test==0].flatten(), y_test[y_test==1].flatten()], 
                    bins=2, alpha=0.7, label=['Bad (Real)', 'Good (Real)'], color=['red', 'green'])
    axes[0, 2].hist([y_pred_test[y_pred_test==0].flatten(), y_pred_test[y_pred_test==1].flatten()], 
                    bins=2, alpha=0.5, label=['Bad (Pred)', 'Good (Pred)'], color=['darkred', 'darkgreen'])
    axes[0, 2].set_title('Distribuição - Conjunto de Teste', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Classe')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].legend()
    
    # 4. Matriz de Confusão - Treino
    def confusion_matrix_manual(y_true, y_pred):
        """Calcula matriz de confusão manualmente"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[tn, fp], [fn, tp]])
    
    cm_train = confusion_matrix_manual(y_train, y_pred_train)
    im1 = axes[1, 0].imshow(cm_train, interpolation='nearest', cmap='Blues')
    axes[1, 0].set_title('Matriz de Confusão - Treino', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predito')
    axes[1, 0].set_ylabel('Real')
    
    # Adicionar números na matriz
    for i in range(cm_train.shape[0]):
        for j in range(cm_train.shape[1]):
            axes[1, 0].text(j, i, str(cm_train[i, j]), ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 5. Matriz de Confusão - Teste
    cm_test = confusion_matrix_manual(y_test, y_pred_test)
    im2 = axes[1, 1].imshow(cm_test, interpolation='nearest', cmap='Blues')
    axes[1, 1].set_title('Matriz de Confusão - Teste', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Predito')
    axes[1, 1].set_ylabel('Real')
    
    # Adicionar números na matriz
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            axes[1, 1].text(j, i, str(cm_test[i, j]), ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 6. Comparação de Acurácias
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    
    categories = ['Treino', 'Teste']
    accuracies = [train_acc, test_acc]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[1, 2].bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 2].set_title('Comparação de Acurácias', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Acurácia')
    axes[1, 2].set_ylim(0, 1.1)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('nba_mlp_resultados.png', dpi=200, bbox_inches='tight')
    print("Gráfico salvo como 'nba_mlp_resultados.png'")
    plt.close()

def main():
    """Função principal"""
    print("=== MLP para Classificação de Jogadores NBA ===\n")
    
    # Definir seed para reprodutibilidade
    np.random.seed(42)
    
    # Carregar e preprocessar dados
    X, y, feature_names, df = load_and_preprocess_nba_data()
    
    print(f"\nDados preprocessados:")
    print(f"  - Shape X: {X.shape}")
    print(f"  - Shape y: {y.shape}")
    print(f"  - Proporção Good: {np.mean(y):.3f}")
    print(f"  - Features: {len(feature_names)}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    print(f"\nDivisão dos dados:")
    print(f"  - Treino: {X_train.shape[0]} amostras")
    print(f"  - Teste: {X_test.shape[0]} amostras")
    
    # Criar e treinar MLP
    print(f"\nCriando MLP...")
    n_features = X_train.shape[1]
    mlp = MLP(layers=[n_features, 16, 8, 1], learning_rate=0.01)
    
    print(f"Arquitetura: {mlp.layers}")
    print(f"Treinando por 1500 épocas...")
    
    # Treinar
    losses = mlp.train(X_train, y_train, epochs=1500, verbose=True)
    
    # Fazer predições
    print(f"\nFazendo predições...")
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)
    
    y_prob_train = mlp.predict_proba(X_train)
    y_prob_test = mlp.predict_proba(X_test)
    
    # Calcular métricas
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    
    print(f"\n=== RESULTADOS DA MLP ===")
    print(f"Arquitetura da Rede:")
    print(f"  - Camadas: {len(mlp.layers)} ({mlp.layers})")
    print(f"  - Camada de Entrada: {mlp.layers[0]} neurônios")
    print(f"  - Camadas Ocultas: {len(mlp.layers)-2} camadas {mlp.layers[1:-1]}")
    print(f"  - Camada de Saída: {mlp.layers[-1]} neurônio")
    print(f"  - Total de Parâmetros: {sum(w.size for w in mlp.weights) + sum(b.size for b in mlp.biases)}")
    print(f"  - Taxa de Aprendizado: {mlp.learning_rate}")
    print(f"  - Épocas de Treinamento: 1500")
    print(f"\nDesempenho:")
    print(f"  - Acurácia Treino: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
    print(f"  - Acurácia Teste:  {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"  - Loss Final:      {losses[-1]:.4f}")
    print(f"  - Loss Inicial:    {losses[0]:.4f}")
    print(f"  - Redução do Loss: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    # Mostrar alguns exemplos de predições
    print(f"\n=== EXEMPLOS DE PREDIÇÕES (Teste) ===")
    for i in range(min(10, len(y_test))):
        real = "Good" if y_test[i][0] == 1 else "Bad"
        pred = "Good" if y_pred_test[i][0] == 1 else "Bad"
        prob = y_prob_test[i][0]
        status = "✓" if y_test[i][0] == y_pred_test[i][0] else "✗"
        print(f"{status} Real: {real:4s} | Pred: {pred:4s} | Prob: {prob:.3f}")
    
    # Criar visualizações
    print(f"\nCriando visualizações...")
    plot_results(losses, y_train, y_pred_train, y_test, y_pred_test, feature_names)
    
    print(f"\n✅ Análise MLP NBA concluída!")
    print(f"   - Dataset: {X.shape[0]} jogadores NBA")
    print(f"   - Features: {len(feature_names)} estatísticas")
    print(f"   - Arquitetura: {mlp.layers} ({len(mlp.layers)} camadas)")
    print(f"   - Parâmetros: {sum(w.size for w in mlp.weights) + sum(b.size for b in mlp.biases)}")
    print(f"   - Acurácia final: {test_accuracy*100:.1f}%")
    print(f"   - Gráfico salvo: nba_mlp_resultados.png")

if __name__ == "__main__":
    main()