import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_and_save_model(df_features, target_column, model_path='models/model.pkl'):
    # Separar X e y
    feature_cols = ['similaridade_skills', 'match_ingles', 'match_academico']
    X = df_features[feature_cols]
    y = df_features[target_column]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Avaliação rápida
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Importância das features
    importances = model.feature_importances_
    feature_names = ['similaridade_skills', 'match_ingles', 'match_academico']
    print(\"\\nImportância das features:\")
    for feat, imp in zip(feature_names, importances):
        print(f'{feat}: {imp:.3f}')


    # Salva modelo
    joblib.dump(model, model_path)
    print(f"Modelo salvo em {model_path}")

if __name__ == "__main__":
    df = pd.read_parquet('dados/trusted/features.parquet')
    train_and_save_model(df, target_column='match')
