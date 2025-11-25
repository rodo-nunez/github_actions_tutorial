import sys
import pandas as pd
import seaborn as sns
import os
from urllib.error import HTTPError

def validar_dataset():
    """Valida que el dataset cumpla con los requisitos mínimos."""
    print("🔍 Iniciando validación del dataset de penguins...")
    try:
        # Cargar datos con manejo de error HTTP
        try:
            df = sns.load_dataset('penguins')
        except HTTPError as e:
            if e.code == 429:
                print("⚠️  Rate limit alcanzado. Intentando cargar desde caché local...")
                # Intentar cargar desde el directorio de caché de seaborn
                cache_dir = 'data/'
                cache_file = os.path.join(cache_dir, 'penguins.csv')
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file)
                    print("✓ Dataset cargado desde caché local")
                else:
                    print("❌ No se pudo cargar el dataset. Descarga el archivo manualmente de:")
                    print("   https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
                    print(f"   y guárdalo en: {cache_file}")
                    return False
            else:
                raise
        
        print(
            f"✓ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
        # Validación 1: Columnas esperadas
        columnas_esperadas = [
            'species', 'island', 'bill_length_mm',
            'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'sex'
        ]
        for col in columnas_esperadas:
            if col not in df.columns:
                print(f"❌ Columna faltante: {col}")
                return False
        print(f"✓ Todas las columnas esperadas presentes")
        # Validación 2: Verificar cantidad mínima de datos
        if len(df) < 300:
            print(f"❌ Dataset muy pequeño: {len(df)} filas (mínimo 300)")
            return False
        print(f"✓ Tamaño adecuado: {len(df)} filas")
        # Validación 3: Verificar especies
        especies_esperadas = {'Adelie', 'Chinstrap', 'Gentoo'}
        especies_encontradas = set(df['species'].dropna().unique())
        if especies_esperadas != especies_encontradas:
            print(
                f"❌ Especies incorrectas. Esperadas: {especies_esperadas}, Encontradas: {especies_encontradas}")
            return False
        print(f"✓ Especies correctas: {especies_encontradas}")
        # Validación 4: Rangos de valores numéricos
        validaciones_rangos = {
            'bill_length_mm': (30, 60),
            'bill_depth_mm': (10, 25),
            'flipper_length_mm': (170, 240),
            'body_mass_g': (2500, 6500)
        }
        for columna, (min_val, max_val) in validaciones_rangos.items():
            valores = df[columna].dropna()
            if valores.min() < min_val or valores.max() > max_val:
                print(
                    f"❌ {columna} fuera de rango [{min_val}, {max_val}]: min={valores.min():.1f}, max={valores.max():.1f}")
                return False
        print("✓ Rangos de valores numéricos válidos")
        # Validación 5: Porcentaje de valores faltantes
        porcentaje_nulos = (df.isnull().sum().sum() /
                            (len(df) * len(df.columns))) * 100
        if porcentaje_nulos > 5:
            print(f"❌ Demasiados valores faltantes: {porcentaje_nulos:.2f}%")
            return False
        print(f"✓ Valores faltantes aceptables: {porcentaje_nulos:.2f}%")
        print("\n✅ Todas las validaciones pasaron exitosamente!")
        return True
    except Exception as e:
        print(f"❌ Error durante la validación: {str(e)}")
        return False


if __name__ == '__main__':
    exito = validar_dataset()
    sys.exit(0 if exito else 1)
