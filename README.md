[![CI](https://github.com/your-repo/atlas_v2/actions/workflows/ci.yml/badge.svg)]
# ATLAS v2

Assistant de trading semi-automatique pour XAU/USD : le programme propose un trade, tu valides.

## Installation (Windows)
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py -m pytest
```
Résultat attendu : `36 passed`.

## Contrôler tes données réelles
1. Dans MT5 : Affichage > Symboles > onglet Barres > XAUUSD > H1 > Exporter.
2. Copie le fichier dans `data/raw/XAUUSD_H1.csv`.
3. Lance :
```
py scripts/check_data.py data/raw/XAUUSD_H1.csv --tf 60
```
Si le chargement échoue, colle-moi les 3 premières lignes du fichier.

## Structure
- `atlas/config.py` : specs XAU/USD (à vérifier avec ton broker) et risque max
- `atlas/data/` : chargement et validation des chandeliers
- `atlas/indicators.py` : EMA, ATR (sans accès au futur, testés)
- `atlas/risk.py` : taille de position en lots, arrondie toujours vers le bas
- `tests/` : 36 tests

## Règles
- Un commit par session, dépôt GitHub PRIVE.
- Jamais de `.env` ni de clés dans Git.
- Aucun backtest sur des données qui n'ont pas passé `check_data.py`.

## Prochaine étape
Backtest minimal : entrée, SL/TP, frais, spread, gaps, PnL juste.
