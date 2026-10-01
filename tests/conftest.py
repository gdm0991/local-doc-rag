import os
import sys
from pathlib import Path

# Тесты не должны качать модель из сети: векторный индекс в тестах — TF-IDF.
os.environ.setdefault("LDR_EMBED_BACKEND", "tfidf")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
