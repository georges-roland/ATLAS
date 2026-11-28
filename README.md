# ATLAS v2

Assistant de trading semi-automatique pour XAU/USD : le programme propose un trade, tu valides.
Mode démo uniquement tant que les données réelles et le paper trading ne sont pas validés.

## Installation (Windows)
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py -m pytest
```
Résultat attendu : `100 passed`, sans avertissement.

## Données (étape obligatoire)
1. Exporte l'historique brut depuis MT5 (Ctrl+U > Barres > XAUUSD > H1 > Demander > Exporter).
2. Enregistre-le sous `data/raw/XAUUSD_H1.csv` (ne l'ouvre pas avec Excel).
3. Contrôle-le :
```
py scripts/check_data.py data/raw/XAUUSD_H1.csv --tf 60
```
Le programme refuse un fichier avec des bougies de week-end, des doublons, des OHLC incohérents, ou trop de bougies plates/copiées.

**Règle : aucune donnée ne doit être créée ou complétée par une IA. Les trous de week-end sont normaux.**

## Backtest
```
py scripts/backtest.py data/raw/XAUUSD_H1.csv --short --spread 0.3 --commission 7
```
Choisir les réglages sur une période, vérifier sur une autre (jamais l'inverse) :
```
py scripts/backtest.py data/raw/XAUUSD_H1.csv --short --end 2024-12-31
py scripts/backtest.py data/raw/XAUUSD_H1.csv --short --start 2025-01-01
```
`--start` : les indicateurs repartent de zéro à cette date (chauffe de quelques bougies sans trade).
Remplace spread, slippage et commission par ceux de ton broker.
Moins de 30 trades : le résultat n'est pas significatif (le programme l'indique).
Compare toujours au résultat « achat-conservation » affiché en bas.

## Validation hors échantillon
Teste 9 réglages (EMA 10/20/50 x stop 1,0/1,5/2,0 ATR) sur une période de choix, puis le meilleur sur la période suivante :
```
py scripts/train_test.py data/raw/XAUUSD_H1.csv --train-end 2024-12-31 --short --spread 0.3 --commission 7
```
Le verdict dit si le meilleur réglage tient hors échantillon, s'il bat l'achat-conservation, ou s'il y a trop peu de trades pour conclure (moins de 30). Un résultat positif n'est qu'une première vérification, pas une preuve.

## Structure
- `atlas/config.py` : specs XAU/USD (à vérifier avec ton broker)
- `atlas/data/` : chargement (UTF-8/UTF-16) et validation des chandeliers
- `atlas/indicators.py` : EMA, ATR
- `atlas/risk.py` : taille de position en lots, arrondie vers le bas
- `atlas/backtest.py` : exécution réaliste, coûts, vrai drawdown, résultats par année
- `atlas/validation.py` : choix des réglages sur une période, vérification sur la suivante
- `scripts/` : `check_data.py`, `backtest.py`, `train_test.py`
- `tests/` : 100 tests
- `legacy/` : ancienne interface Flask, non maintenue (incompatible avec le nouveau backtest)

## Règles
- Dépôt GitHub PRIVE. Jamais de `.env`, clé ou token dans Git.
- Un commit par session ; lance `py -m pytest` avant chaque commit.
- Aucun backtest sur des données qui n'ont pas passé `check_data.py`.
- Un résultat n'est vrai que s'il sort de TON terminal.
