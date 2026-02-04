#!/usr/bin/env bash
# make_all.sh
# Exécute les scripts du TP et range toutes les figures .png dans ./figures
#
# IMPORTANT :
# - Modifiez la variable BASE_DIR ci-dessous si vous avez placé le projet ailleurs
#   (ex: /home/votre_user/TP1_RN_deep_debache).

set -euo pipefail

# >>> A MODIFIER SI BESOIN <<<
BASE_DIR="/home/ubuntu/Videos/TP1_RN_deep_debache"

cd "$BASE_DIR"

echo "=== TP1 MNIST : génération de toutes les figures (python3.10) ==="
echo "Dossier de travail : $BASE_DIR"

# Backend matplotlib non-interactif (utile sans interface graphique)
export MPLBACKEND=Agg
# Réduire les logs TensorFlow (optionnel)
export TF_CPP_MIN_LOG_LEVEL=2

# Dossier de sortie figures (créé dans BASE_DIR)
FIG_DIR="$BASE_DIR/figures"
mkdir -p "$FIG_DIR"

# Vérifier que les scripts existent
for f in exo0.py exo1.py mlp.py; do
  if [[ ! -f "$BASE_DIR/$f" ]]; then
    echo "ERREUR: fichier manquant: $BASE_DIR/$f"
    echo "=> Placez $f dans ce dossier ou adaptez BASE_DIR/les chemins dans make_all.sh"
    exit 1
  fi
done

echo "[1/3] Exo0 : visualisation des 200 images"
python3.10 "$BASE_DIR/exo0.py"

echo "[2/3] Exo1 : régression logistique (loss/acc + templates W)"
python3.10 "$BASE_DIR/exo1.py"

echo "[3/3] Exo2 : MLP (init 0 / normal / xavier) + courbes"
python3.10 "$BASE_DIR/mlp.py"

echo
echo "=== Déplacement des figures .png vers $FIG_DIR ==="

shopt -s nullglob
PNG_FILES=( "$BASE_DIR"/*.png )
if (( ${#PNG_FILES[@]} > 0 )); then
  mv -f "${PNG_FILES[@]}" "$FIG_DIR/"
  echo "OK : ${#PNG_FILES[@]} fichier(s) déplacé(s)."
else
  echo "Aucun fichier .png trouvé à déplacer."
fi
shopt -u nullglob

echo
echo "=== Terminé. Figures disponibles dans : $FIG_DIR ==="
ls -1 "$FIG_DIR" 2>/dev/null || true
