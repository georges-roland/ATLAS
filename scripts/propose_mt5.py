"""Affiche une proposition MT5 en lecture seule ; n'envoie aucun ordre."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.mt5_live import evaluate_mt5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument(
        "--costs-verified",
        action="store_true",
        help="Confirme que les hypothèses de spread et de slippage correspondent au broker.",
    )
    parser.add_argument(
        "--commission-per-lot-account-currency",
        type=float,
        help="Commission totale entrée + sortie par lot, dans la devise du compte.",
    )
    args = parser.parse_args()

    result = evaluate_mt5(
        symbol=args.symbol,
        costs_verified=args.costs_verified,
        commission_per_lot_account_currency=args.commission_per_lot_account_currency,
    )
    print(f"Serveur démo : {result.server}")
    print(f"Symbole : {result.symbol}")
    print(
        f"Compte : {result.equity:.2f} {result.account_currency} ; "
        f"positions ouvertes : {result.position_count}"
    )
    print(f"Spread actuel : {result.spread:.5f}")
    print(
        f"Bougies H4 clôturées : {result.closed_candle_count} ; "
        f"dernière : {result.last_closed_candle_utc.isoformat()}"
    )
    if result.signal is None:
        print("Signal : aucun")
    else:
        print(
            f"Signal : {result.signal.action} ; "
            f"clôture de référence : {result.signal.reference_price:.3f} ; "
            f"heure UTC : {result.signal.time.isoformat()}"
        )
    if result.proposal is None and result.signal is None:
        print("Proposition : aucune (aucun signal)")
    elif result.proposal is None:
        print(f"Proposition : bloquée ({result.blocked_reason or 'aucun signal d’achat'})")
    else:
        proposal = result.proposal
        print("Proposition : en attente de validation humaine")
        print(
            f"Entrée : {proposal.entry_price:.3f} ; stop : {proposal.stop_price:.3f} ; "
            f"lots : {proposal.lots:.2f} ; risque estimé : {proposal.risk_amount:.2f} "
            f"{result.account_currency}"
        )


if __name__ == "__main__":
    main()
