cat > README.md << 'EOF'
# TP1 — Algorithme de rétro-propagation de l’erreur (MNIST)

Implémentation **from scratch (NumPy)** de :
- Régression logistique multi-classes (softmax + entropie croisée)
- Perceptron Multi-Couches (MLP) à 1 couche cachée (sigmoïde + softmax)
sur la base **MNIST**.

## Structure du projet

- `scripts/`
  - `exo0.py` : visualisation des 200 premières images MNIST
  - `exo1.py` : régression logistique (SGD mini-batch) + courbes + figures
  - `mlp.py`  : MLP (SGD mini-batch) + comparaison initialisations (zéro / normal / Xavier)
  - `make_all.sh` : lance tous les scripts et génère les sorties
- `figures/` : figures générées (loss, accuracy, matrices de poids, etc.)
- `PDF/TP1_RN_deep_debache.pdf` : rapport final
- `TP1.ipynb` : notebook (si utilisé)
- `requirements.txt` : dépendances

## Prérequis

- Ubuntu
- Python 3.10

Installation des dépendances :

```bash
python3.10 -m pip install -r requirements.txt
# TP1_RN_deep_debache
