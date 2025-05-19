
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
from scipy.stats import randint, uniform

def load_and_preprocess(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"✅ Dados carregados: {df.shape}")

    # Feature engineering
    df['composite_score'] = 0.5 * df['skills_similarity'] + \
                            0.3 * df['area_match'] + \
                            0.2 * df['tfidf_similarity']

    X = df.drop(columns=['match', 'codigo_profissional', 'vaga_id'], errors='ignore')
    y = df['match']

    # Visualização da separação das classes
    print("\n📊 Visualizando distribuição das features por classe...")
    df_viz = X.copy()
    df_viz["match"] = y
    sns.pairplot(df_viz, hue="match", plot_kws={'alpha': 0.3})
    plt.suptitle("Distribuição das features", y=1.02)
    plt.savefig("visuals/pairplot_features.png")
    plt.close()

    return X, y

def train_with_random_search(X_train, y_train):
    print("\n🔍 Iniciando RandomizedSearchCV com XGBoost...")

    xgb = XGBClassifier(
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )

    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(2, 6),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }

    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        scoring='f1', n_iter=15, cv=3, verbose=1,
        random_state=42, n_jobs=-1
    )

    search.fit(X_train, y_train)
    print(f"✅ Melhor configuração: {search.best_params_}")
    return search.best_estimator_

def plot_feature_importance(model, feature_names):
    print("\n📊 Gerando gráfico de importância das features...")
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    plt.barh(feature_names, importances)
    plt.title("Importância das features")
    plt.tight_layout()
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    try:
        print("🔧 Iniciando pipeline otimizado...")

        os.makedirs("visuals", exist_ok=True)
        X, y = load_and_preprocess("dados/trusted/features.parquet")

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Aplicar SMOTE
        print("\n⚖️ Aplicando SMOTE para balancear classes...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"🔢 Novo shape do treino balanceado: {X_train_bal.shape}")

        # Treinar modelo com RandomizedSearchCV
        model = train_with_random_search(X_train_bal, y_train_bal)

        # Avaliação
        y_prob = model.predict_proba(X_test)[:, 1]

        # Otimização de threshold
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        print(f"\n🔧 Melhor threshold encontrado: {best_threshold:.4f}")
        y_pred_best = (y_prob >= best_threshold).astype(int)

        print("\n📊 Classification Report (threshold otimizado):")
        print(classification_report(y_test, y_pred_best))

        print("\n🧱 Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred_best))

        print(f"\n📈 ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

        # Feature importance
        plot_feature_importance(model, X.columns)

        # Salvar modelo
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fast_model.pkl")
        print("💾 Modelo salvo em: models/fast_model.pkl")

        print("\n🎯 Pipeline concluído com sucesso!")

    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
