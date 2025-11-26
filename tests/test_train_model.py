"""
Tests para el módulo de entrenamiento del modelo.
"""
from src.train_model import cargar_datos, preparar_datos, entrenar_modelo
import pandas as pd
import sys

# Agregar src al path
sys.path.insert(0, 'src')


def test_cargar_datos():
    """Test que verifica que los datos se cargan correctamente."""
    df = cargar_datos()

    # Verificar que es un DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verificar que no está vacío
    assert len(df) > 0

    # Verificar que tiene las columnas esperadas
    columnas_esperadas = ['bill_length_mm', 'bill_depth_mm',
                          'flipper_length_mm', 'body_mass_g', 'species']
    for col in columnas_esperadas:
        assert col in df.columns

    # Verificar que no hay nulos
    assert df.isnull().sum().sum() == 0


def test_preparar_datos():
    """Test que verifica la preparación de datos."""
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    # Verificar tamaños
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Verificar que train + test = total
    assert len(X_train) + len(X_test) == len(df)

    # Verificar que X tiene 4 features
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4


def test_entrenar_modelo():
    """Test que verifica que el modelo entrena correctamente."""
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    modelo = entrenar_modelo(X_train, y_train)

    # Verificar que el modelo tiene el método predict
    assert hasattr(modelo, 'predict')

    # Verificar que puede hacer predicciones
    predicciones = modelo.predict(X_test)
    assert len(predicciones) == len(X_test)

    # Verificar que la accuracy es razonable (> 70%)
    accuracy = (predicciones == y_test).mean()
    assert accuracy > 0.7, f"Accuracy muy baja: {accuracy:.3f}"


def test_especies_correctas():
    """Test que verifica que las especies predichas son válidas."""
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    modelo = entrenar_modelo(X_train, y_train)
    predicciones = modelo.predict(X_test)

    # Verificar que solo predice las especies conocidas
    especies_validas = {'Adelie', 'Chinstrap', 'Gentoo'}
    especies_predichas = set(predicciones)

    assert especies_predichas.issubset(especies_validas)


def test_reproducibilidad():
    """Test que verifica que el modelo es reproducible."""
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    # Entrenar dos modelos con la misma semilla
    modelo1 = entrenar_modelo(X_train, y_train)
    modelo2 = entrenar_modelo(X_train, y_train)

    # Las predicciones deben ser idénticas
    pred1 = modelo1.predict(X_test)
    pred2 = modelo2.predict(X_test)

    assert all(pred1 == pred2), "El modelo no es reproducible"
