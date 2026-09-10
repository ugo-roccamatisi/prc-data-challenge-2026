"""Point d'entree unique : reproduit la soumission finale.

Prerequis :
  1. pip install -r requirements.txt
  2. Telecharger les donnees du challenge dans ./data/ (voir README, section Data)

Le pipeline (src/pipeline.py) : entrainement, validation temporelle sur
janvier + juillet 2025, reentrainement sur 2025 complet, et ecriture des
fichiers de soumission dans ./submissions/.

Premiere execution : ~20-40 min (calcul des features, mise en cache), puis
~10-15 min par relance.
"""
import runpy
from pathlib import Path

if not any(Path("data").glob("training_*.parquet")):
    raise SystemExit("Donnees absentes : voir README, section Data.")

runpy.run_path(Path("src") / "pipeline.py", run_name="__main__")
