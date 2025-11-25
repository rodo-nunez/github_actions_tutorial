"""
Script simple para entrenar un modelo de clasificación con el dataset Penguins.
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path


def cargar_datos():
    """Carga el dataset de penguins desde seaborn."""
    df = sns.load_dataset('penguins')
    df = df.dropna()  # Eliminar valores faltantes
    return df


def preparar_datos(df):
    """Prepara los datos para el entrenamiento."""
    # Features: características numéricas
    X = df[['bill_length_mm', 'bill_depth_mm',
            'flipper_length_mm', 'body_mass_g']]

    # Target: especie del pingüino
    y = df['species']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def entrenar_modelo(X_train, y_train):
    """Entrena un clasificador Random Forest simple."""
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    """Evalúa el modelo y retorna métricas."""
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy: {accuracy:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


def guardar_modelo(modelo, ruta='models/penguin_classifier.pkl'):
    """Guarda el modelo entrenado."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)

    with open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

    print(f"\n✅ Modelo guardado en: {ruta}")


def main():
    print("🐧 Entrenando clasificador de especies de pingüinos...")

    # Pipeline completo
    df = cargar_datos()
    print(f"✓ Datos cargados: {len(df)} muestras")

    X_train, X_test, y_train, y_test = preparar_datos(df)
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

    modelo = entrenar_modelo(X_train, y_train)
    print("✓ Modelo entrenado")

    accuracy = evaluar_modelo(modelo, X_test, y_test)

    guardar_modelo(modelo)

    return accuracy


if __name__ == '__main__':
    main()
