import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
from scipy.stats import randint, uniform
import json
from datetime import datetime
import mlflow
from mlflow.models.signature import infer_signature
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')
pd.set_option('display.max_columns', 50)


class AdvancedMLPipeline:
    def __init__(self):
        
        self.experiment_name = "MonitoringJobMatching"
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = "models"
        self.metrics_dir = "metrics"
        self.artifacts_dir = "artifacts"
        self._create_directories()

        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=f"run_{self.run_id}")

    def _create_directories(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs("visuals", exist_ok=True)

    def load_data(self, parquet_path):
        print("⏳ Carregando dados.")
        start = datetime.now()

        df = pd.read_parquet(parquet_path)

        cols_to_load = [
            'skills_similarity', 'area_match', 'tfidf_similarity',
            'n_cv_skills', 'n_job_skills', 'skills_coverage_ratio',
            'keyword_overlap', 'match', 'codigo_profissional', 'vaga_id'
        ] + [col for col in df.columns if col.startswith('senioridade_')]

        df = df[cols_to_load]

        df['composite_score'] = (
            0.5 * df['skills_similarity'].astype('float32') +
            0.3 * df['area_match'].astype('float32') +
            0.2 * df['tfidf_similarity'].astype('float32')
        )

        X = df.drop(columns=['match', 'codigo_profissional', 'vaga_id'], errors='ignore')
        y = df['match'].astype('int8')

        mlflow.log_param("data_shape", df.shape)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("class_ratio", y.mean())

        print(f"✅ Dados carregados: {df.shape} [{(datetime.now()-start).total_seconds():.2f}s]")
        return X, y

    def generate_visualizations(self, X, y):
        try:
            print("\n📊 Gerando visualizações.")
            plt.close('all')

            sample_size = min(5000, len(X))
            df_viz = X.sample(n=sample_size, random_state=42)
            df_viz['match'] = y.loc[df_viz.index]

            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df_viz, x='composite_score', hue='match', common_norm=False)
            plt.title('Distribuição do Composite Score')
            plt.savefig("visuals/composite_dist.png", dpi=100, bbox_inches='tight')
            mlflow.log_artifact("visuals/composite_dist.png")
            plt.close()

            plt.figure(figsize=(12, 8))
            sns.heatmap(df_viz.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Matriz de Correlação')
            plt.savefig("visuals/correlation_matrix.png", dpi=100, bbox_inches='tight')
            mlflow.log_artifact("visuals/correlation_matrix.png")
            plt.close()

            print("✅ Visualizações geradas com sucesso")
        except Exception as e:
            print(f"⚠️ Erro nas visualizações: {e}")

    def train_model(self, X_train, y_train):
        print("\n🔍 Iniciando treinamento com RandomizedSearchCV.")
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
            'scale_pos_weight': [1, y_train.mean()]
        }

        search = RandomizedSearchCV(
            xgb, param_distributions=param_dist,
            scoring='f1', n_iter=20, cv=3, verbose=2,
            random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)

        mlflow.log_params({k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_score", search.best_score_)

        print(f"✅ Treino concluído")
        print(f"Melhores parâmetros: {search.best_params_}")
        return search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        print("\n📈 Avaliando modelo.")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        y_pred_opt = (y_prob >= best_threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1_score": f1_score(y_test, y_pred),
            "f1_score_opt": f1_score(y_test, y_pred_opt),
            "best_threshold": float(best_threshold),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "confusion_matrix_opt": confusion_matrix(y_test, y_pred_opt).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        mlflow.log_metrics({
            "roc_auc": metrics["roc_auc"],
            "f1_score": metrics["f1_score"],
            "f1_score_opt": metrics["f1_score_opt"],
            "best_threshold": metrics["best_threshold"]
        })

        with open(os.path.join(self.metrics_dir, f"metrics_{self.run_id}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(os.path.join(self.metrics_dir, f"metrics_{self.run_id}.json"))

        print(f"\n🔧 Melhor threshold: {best_threshold:.4f}")
        print("\n📊 Classification Report (threshold padrão 0.5):")
        print(classification_report(y_test, y_pred))
        print("\n📊 Classification Report (threshold otimizado):")
        print(classification_report(y_test, y_pred_opt))

        return metrics

    def analyze_errors(self, model, X_test, y_test):
        print("\n🔎 Analisando erros.")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        errors = X_test.copy()
        errors['true'] = y_test
        errors['pred'] = y_pred
        errors['prob'] = y_prob

        fp = errors[(errors['true'] == 0) & (errors['pred'] == 1)]
        fn = errors[(errors['true'] == 1) & (errors['pred'] == 0)]

        error_analysis = {
            "false_positives_stats": fp.describe().to_dict(),
            "false_negatives_stats": fn.describe().to_dict(),
            "n_false_positives": int(len(fp)),
            "n_false_negatives": int(len(fn)),
            "fp_mean_prob": float(fp['prob'].mean()) if not fp.empty else 0.0,
            "fn_mean_prob": float(fn['prob'].mean()) if not fn.empty else 0.0
        }

        mlflow.log_metrics({
            "n_false_positives": error_analysis["n_false_positives"],
            "n_false_negatives": error_analysis["n_false_negatives"]
        })

        with open(os.path.join(self.artifacts_dir, f"error_analysis_{self.run_id}.json"), 'w') as f:
            json.dump(error_analysis, f, indent=2)
        mlflow.log_artifact(os.path.join(self.artifacts_dir, f"error_analysis_{self.run_id}.json"))

        print("✅ Análise de erros concluída")
        return error_analysis

    def save_model(self, model, X_sample):
        print("\n💾 Salvando modelo.")
        signature = infer_signature(X_sample, model.predict(X_sample))
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_sample.iloc[:5],
            registered_model_name="JobMatchingModel"
        )

        model_path = os.path.join(self.model_dir, f"model_{self.run_id}.pkl")
        joblib.dump(model, model_path)
        self.plot_feature_importance(model, X_sample.columns)
        print(f"✅ Modelo salvo em: {model_path}")
        return model_path

    def plot_feature_importance(self, model, feature_names):
        try:
            print("\n📊 Gerando importância de features.")
            importances = model.feature_importances_
            top_n = min(15, len(feature_names))
            sorted_idx = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), importances[sorted_idx])
            plt.yticks(range(top_n), np.array(feature_names)[sorted_idx])
            plt.title(f"Top {top_n} Features por Importância")
            plt.tight_layout()

            path = "visuals/feature_importance.png"
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(path)
            print("✅ Gráfico de importância salvo")
        except Exception as e:
            print(f"⚠️ Erro ao gerar importância: {str(e)}")

    def run_pipeline(self, data_path):
        try:
            start = datetime.now()

            X, y = self.load_data(data_path)
            self.generate_visualizations(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            print("\n⚖️ Aplicando SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"🔢 Dimensões pós-balanceamento: {X_train_bal.shape}")

            model = self.train_model(X_train_bal, y_train_bal)
            metrics = self.evaluate_model(model, X_test, y_test)
            self.analyze_errors(model, X_test, y_test)
            model_path = self.save_model(model, X_test)

            print(f"\n🎯 Pipeline concluído em {(datetime.now()-start).total_seconds()/60:.1f} minutos.")
            return {
                "model_path": model_path,
                "metrics": metrics,
                "run_id": self.run_id
            }

        except Exception as e:
            print(f"\n❌ Erro no pipeline: {str(e)}")
            raise
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    start = datetime.now()
    pipeline = AdvancedMLPipeline()
    results = pipeline.run_pipeline("dados/trusted/features.parquet")
    print(f"\n🚀 Tempo total de execução: {(datetime.now()-start).total_seconds()/60:.1f} minutos")
    print(f"📌 Run ID: {results['run_id']}")
    print(f"💾 Modelo salvo em: {results['model_path']}")
