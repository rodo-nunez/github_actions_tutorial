# Proyecto de Prueba - GitHub Actions

Proyecto simple para testear todos los workflows de GitHub Actions del tutorial.

## 🐧 Descripción

Este proyecto entrena un modelo de clasificación básico usando el dataset de **Palmer Penguins** para predecir especies de pingüinos basándose en características físicas.

## 📁 Estructura del Proyecto

```
.
├── .github/workflows/     # Workflows de GitHub Actions
│   ├── hello-world.yml    # Workflow básico de prueba
│   ├── tests.yml          # Tests automáticos (Python 3.9-3.11)
│   ├── validate-data.yml  # Validación de datos en PRs
│   └── train-model.yml    # Entrenamiento automático nocturno
├── src/
│   └── train_model.py     # Script principal del modelo
├── tests/
│   └── test_train_model.py # Tests unitarios
├── scripts/
│   ├── train.py           # Script para entrenar el modelo
│   └── validar_datos.py   # Script de validación de datos
└── pyproject.toml         # Configuración de dependencias
```

## 🚀 Instalación

Este proyecto usa `uv` para manejar dependencias de Python:

```bash
# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias
uv pip install -e .
```

## 🧪 Uso Local

### Entrenar el modelo

```bash
python scripts/train.py
```

### Ejecutar tests

```bash
pytest tests/ -v
```

### Validar datos

```bash
python scripts/validar_datos.py
```

## 🤖 GitHub Actions Workflows

### 1. **Hello World** (`hello-world.yml`)
- Se ejecuta en cada push
- Workflow básico de prueba

### 2. **Tests** (`tests.yml`)
- Se ejecuta en push y PRs
- Matriz de Python 3.9, 3.10, 3.11
- Ejecuta todos los tests unitarios

### 3. **Validación de Datos** (`validate-data.yml`)
- Se ejecuta solo en PRs
- Verifica integridad del dataset
- Genera reporte de validación

### 4. **Entrenamiento de Modelo** (`train-model.yml`)
- Se ejecuta diariamente a las 2 AM UTC
- También ejecutable manualmente (workflow_dispatch)
- Entrena el modelo y guarda artefactos
- Guarda modelo entrenado y métricas

## 📊 Modelo

- **Algoritmo**: Random Forest Classifier
- **Dataset**: Palmer Penguins (seaborn)
- **Features**: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
- **Target**: species (Adelie, Chinstrap, Gentoo)
- **Accuracy esperado**: ~95%

## 🧩 Dependencias Principales

- `scikit-learn`: Modelo de ML
- `pandas`: Manipulación de datos
- `seaborn`: Dataset de penguins
- `pytest`: Testing

## 📝 Notas

Este es un proyecto **intencionalmente simple** diseñado para demostrar GitHub Actions workflows. No está optimizado para producción.