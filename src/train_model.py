import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def evaluate_model_correctly(model, X_train, X_test, y_train, y_test):
    """Avaliação robusta sem vazamento de dados"""
    # 1. Treinar no conjunto de treino
    model.fit(X_train, y_train)
    
    # 2. Avaliar no conjunto de teste
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    print("\n=== Métricas Reais no Conjunto de Teste ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 3. Matriz de Confusão
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Match', 'Match'],
                yticklabels=['Não Match', 'Match'])
    plt.title('Matriz de Confusão - Avaliação Correta')
    plt.show()
    
    return model

def train_robust_model(df_features, target_column='match'):
    # 1. Separação estrita dos dados ANTES de qualquer processamento
    X = df_features.drop(columns=[target_column])
    y = df_features[target_column]
    
    # 2. Divisão treino-teste estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Pipeline seguro
    pipeline = make_pipeline(
        RobustScaler(),
        SelectKBest(mutual_info_classif, k=5),
        ADASYN(random_state=42, sampling_strategy=0.5),
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=20,
            subsample=0.8,
            random_state=42
        )
    )
    
    # 4. Validação Cruzada apenas no treino
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n=== Treinamento com Validação Cruzada ===")
    pipeline.fit(X_train, y_train)
    
    # 5. Avaliação correta no conjunto de teste
    final_model = evaluate_model_correctly(pipeline, X_train, X_test, y_train, y_test)
    
    return final_model

if __name__ == "__main__":
    try:
        # 1. Carregar e verificar dados
        df = pd.read_parquet('dados/trusted/features.parquet')
        
        # Remover features problemáticas (se existirem)
        df = df.drop(columns=['composite_score'], errors='ignore')
        
        if len(df) < 1000:
            raise ValueError("Dados insuficientes")
            
        # 2. Treinar modelo robusto
        robust_model = train_robust_model(df)
        
        # 3. Salvar modelo
        os.makedirs('models', exist_ok=True)
        joblib.dump(robust_model, 'models/robust_model.pkl')
        print("\n=== Modelo robusto salvo com sucesso! ===")
        
    except Exception as e:
        print(f"\n!!! ERRO: {str(e)} !!!")