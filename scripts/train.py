"""
Wrapper script para entrenar el modelo (usado por GitHub Actions).
"""
from src.train_model import main
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

if __name__ == '__main__':
    main()
