import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def train_final_solution(df_features, target_column='match'):
    # 1. Análise Exploratória Avançada
    print("\n=== Análise Detalhada ===")
    print("Distribuição Original:")
    print(df_features[target_column].value_counts(normalize=True))
    
    # 2. Engenharia de Features Avançada
    df_features['interaction_feature'] = df_features['similaridade_skills'] * (df_features['match_ingles'] + df_features['match_academico'])
    df_features['composite_score'] = np.sqrt(df_features['similaridade_skills']) * (df_features['match_ingles'] + df_features['match_academico'])
    
    # 3. Seleção de Features
    X = df_features[['similaridade_skills', 'match_ingles', 'match_academico', 
                    'interaction_feature', 'composite_score']]
    y = df_features[target_column]
    
    # 4. Pipeline Definitivo
    pipeline = Pipeline([
        ('scaler', QuantileTransformer(output_distribution='normal')),
        ('feature_selection', SelectKBest(mutual_info_classif, k=3)),
        ('sampler', BorderlineSMOTE(random_state=42, sampling_strategy=0.75, k_neighbors=7)),
        ('clf', HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=7,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=True,
            scoring='balanced_accuracy',
            random_state=42
        ))
    ])
    
    # 5. Validação Cruzada Estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n=== Treinamento com Validação Cruzada ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    
    # 6. Avaliação Detalhada
    print("\n=== Métricas no Conjunto de Teste ===")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 7. Matriz de Confusão Melhorada
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Match', 'Match'],
                yticklabels=['Não Match', 'Match'])
    plt.title('Matriz de Confusão - Solução Final')
    plt.show()
    
    # 8. Feature Importance
    selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
    importance = pipeline.named_steps['clf'].feature_importances_
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=importance, y=selected_features)
    plt.title('Importância das Features Selecionadas')
    plt.show()
    
    return pipeline

if __name__ == "__main__":
    try:
        # 1. Carregar e Verificar Dados
        df = pd.read_parquet('dados/trusted/features.parquet')
        if len(df) < 1000:
            raise ValueError("Dados insuficientes para treinamento")
        
        # 2. Treinar Solução Final
        final_model = train_final_solution(df)
        
        # 3. Salvar Modelo e Metadados
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/final_solution_model.pkl')
        
        print("\n=== Solução final implementada com sucesso! ===")
        print("Modelo salvo em: models/final_solution_model.pkl")
        
    except Exception as e:
        print(f"\n!!! ERRO CRÍTICO: {str(e)} !!!")