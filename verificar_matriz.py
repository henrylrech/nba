"""
Script para verificar se a matriz de confusão está correta
"""

import pandas as pd
import numpy as np
from main import DataProcessor, MLPNeuralNetwork, ModelEvaluator, split_train_test

def verificar_matriz_confusao():
    print("=== VERIFICAÇÃO DETALHADA DA MATRIZ DE CONFUSÃO ===\n")
    
    # Reproduzir exatamente o mesmo processo
    processor = DataProcessor()
    df = processor.load_and_analyze_data('nba_dados_2024.csv')
    df = processor.create_performance_criteria(df)
    X, y, feature_names = processor.prepare_features_and_target(df, use_new_criteria=True)
    
    # Mesma divisão (mesmo random_state)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    X_train_norm, X_test_norm = processor.normalize_features(X_train, X_test)
    
    # Treinar modelo
    model = MLPNeuralNetwork(X_train_norm.shape[1], [20, 10], 1, 0.1, 0.9)
    model.train(X_train_norm, y_train, epochs=1500, verbose=False)
    
    # Fazer predições
    y_test_pred = model.predict_classes(X_test_norm)
    y_test_proba = model.predict(X_test_norm)
    
    print(f"DADOS DO CONJUNTO DE TESTE:")
    print(f"Total de amostras: {len(y_test)}")
    print(f"Real Good (1): {np.sum(y_test == 1)}")
    print(f"Real Bad (0): {np.sum(y_test == 0)}")
    print(f"Predito Good (1): {np.sum(y_test_pred == 1)}")
    print(f"Predito Bad (0): {np.sum(y_test_pred == 0)}")
    
    print(f"\nVERIFICAÇÃO MANUAL DA MATRIZ:")
    
    # Calcular manualmente cada elemento
    tp_manual = 0  # Real=1, Pred=1
    tn_manual = 0  # Real=0, Pred=0
    fp_manual = 0  # Real=0, Pred=1
    fn_manual = 0  # Real=1, Pred=0
    
    print(f"\nAnálise amostra por amostra (primeiras 20):")
    print(f"{'#':>3} {'Real':>6} {'Pred':>6} {'Prob':>8} {'Tipo':>4}")
    print("-" * 35)
    
    for i in range(min(20, len(y_test))):
        real = y_test[i][0]
        pred = y_test_pred[i][0]
        prob = y_test_proba[i][0]
        
        if real == 1 and pred == 1:
            tp_manual += 1
            tipo = "TP"
        elif real == 0 and pred == 0:
            tn_manual += 1
            tipo = "TN"
        elif real == 0 and pred == 1:
            fp_manual += 1
            tipo = "FP"
        elif real == 1 and pred == 0:
            fn_manual += 1
            tipo = "FN"
        
        print(f"{i+1:3d} {real:6d} {pred:6d} {prob:8.3f} {tipo:>4}")
    
    # Calcular para todo o conjunto
    tp_total = np.sum((y_test.flatten() == 1) & (y_test_pred.flatten() == 1))
    tn_total = np.sum((y_test.flatten() == 0) & (y_test_pred.flatten() == 0))
    fp_total = np.sum((y_test.flatten() == 0) & (y_test_pred.flatten() == 1))
    fn_total = np.sum((y_test.flatten() == 1) & (y_test_pred.flatten() == 0))
    
    print(f"\nCÁLCULO MANUAL COMPLETO:")
    print(f"TP (True Positives): {tp_total}")
    print(f"TN (True Negatives): {tn_total}")
    print(f"FP (False Positives): {fp_total}")
    print(f"FN (False Negatives): {fn_total}")
    print(f"Total: {tp_total + tn_total + fp_total + fn_total}")
    
    # Usar a função do código
    evaluator = ModelEvaluator()
    confusion_matrix = evaluator.confusion_matrix(y_test, y_test_pred)
    
    print(f"\nMATRIZ DE CONFUSÃO (função do código):")
    print(f"[[{confusion_matrix[0,0]:3d}, {confusion_matrix[0,1]:3d}]")
    print(f" [{confusion_matrix[1,0]:3d}, {confusion_matrix[1,1]:3d}]]")
    
    print(f"\nInterpretação da matriz:")
    print(f"                 Predito")
    print(f"               Bad  Good")
    print(f"Real    Bad   {confusion_matrix[0,0]:4d}  {confusion_matrix[0,1]:4d}")
    print(f"        Good  {confusion_matrix[1,0]:4d}  {confusion_matrix[1,1]:4d}")
    
    # Verificar se os cálculos batem
    print(f"\nVERIFICAÇÃO:")
    print(f"TP manual: {tp_total}, TP matriz: {confusion_matrix[1,1]} ✓" if tp_total == confusion_matrix[1,1] else f"TP manual: {tp_total}, TP matriz: {confusion_matrix[1,1]} ✗")
    print(f"TN manual: {tn_total}, TN matriz: {confusion_matrix[0,0]} ✓" if tn_total == confusion_matrix[0,0] else f"TN manual: {tn_total}, TN matriz: {confusion_matrix[0,0]} ✗")
    print(f"FP manual: {fp_total}, FP matriz: {confusion_matrix[0,1]} ✓" if fp_total == confusion_matrix[0,1] else f"FP manual: {fp_total}, FP matriz: {confusion_matrix[0,1]} ✗")
    print(f"FN manual: {fn_total}, FN matriz: {confusion_matrix[1,0]} ✓" if fn_total == confusion_matrix[1,0] else f"FN manual: {fn_total}, FN matriz: {confusion_matrix[1,0]} ✗")
    
    # Calcular métricas manualmente
    accuracy_manual = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
    precision_manual = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_manual = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    
    print(f"\nMÉTRICAS CALCULADAS MANUALMENTE:")
    print(f"Acurácia: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")
    print(f"Precisão: {precision_manual:.4f} ({precision_manual*100:.2f}%)")
    print(f"Recall: {recall_manual:.4f} ({recall_manual*100:.2f}%)")
    
    # Mostrar algumas predições específicas para verificar
    print(f"\nAMOSTRAS ESPECÍFICAS PARA VERIFICAÇÃO:")
    print(f"Todas as amostras Good (Real=1):")
    good_indices = np.where(y_test.flatten() == 1)[0]
    for i in good_indices[:10]:  # Mostrar primeiras 10
        real = y_test[i][0]
        pred = y_test_pred[i][0]
        prob = y_test_proba[i][0]
        status = "✓" if real == pred else "✗"
        print(f"  Amostra {i}: Real={real}, Pred={pred}, Prob={prob:.3f} {status}")
    
    print(f"\nTodas as amostras Bad (Real=0) - primeiras 10:")
    bad_indices = np.where(y_test.flatten() == 0)[0]
    for i in bad_indices[:10]:  # Mostrar primeiras 10
        real = y_test[i][0]
        pred = y_test_pred[i][0]
        prob = y_test_proba[i][0]
        status = "✓" if real == pred else "✗"
        print(f"  Amostra {i}: Real={real}, Pred={pred}, Prob={prob:.3f} {status}")
    
    return tp_total, tn_total, fp_total, fn_total

if __name__ == "__main__":
    tp, tn, fp, fn = verificar_matriz_confusao()
    
    print(f"\n" + "="*60)
    print(f"RESULTADO FINAL DA VERIFICAÇÃO:")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    if fp == 0 and fn == 0:
        print(f"✅ CONFIRMADO: 100% de acurácia e precisão!")
        print(f"✅ A matriz de confusão está CORRETA!")
    else:
        print(f"❌ Há erros: FP={fp}, FN={fn}")
    print(f"="*60)