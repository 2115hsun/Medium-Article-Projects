#!/usr/bin/env python3
"""SuperTrend — fully SELF-CONTAINED pipeline. The ONLY thing you provide is your
3-min OHLCV CSVs. Run it and it produces the trained joblib bundle:

  Step 0  Clone the public FFM library + install deps
  Step 1  Copy your CSVs in, build the 3-min FFM feature parquet
  Step 2  Causality audit (batch == streaming, no look-ahead)
  Step 3  PRODUCE — train the selection head + save the joblib bundle  ← the output
  (Step 3 also runs the overfit-driven training loop first, with --validate)

Both the SuperTrend strategy AND the production training are inlined here (the
strategy is a public indicator; training calls the generic FFM produce.train) —
no private repo needed; only the generic FFM library is cloned.

Validation is the overfit-driven TRAINING LOOP (ev.run loop=True): default
walk-forward → VAL→TEST generalization gate → Optuna only if overfit → rerun →
loop → final FULL walk-forward; produce is GATED on it generalizing.

Usage:
    # put your CSVs (ES_3min.csv, NQ_3min.csv, ...) in ./csv  then:
    python3 supertrend_chronos_pipeline.py --csv-dir ./csv               # → joblib bundle
    python3 supertrend_chronos_pipeline.py --csv-dir ./csv --validate    # + overfit-driven loop first
    python3 supertrend_chronos_pipeline.py --csv-dir ./csv --no-train    # validation only, no bundle
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = 'https://github.com/johnamcruz/Futures-Foundation-Model.git'
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']
TF = '3min'                            # set from --tf in main(); 3min = production

# ── strategy / training constants (the exact ones used) ──────────────────────
DATA_MONTHS = 72
CTX = 128
ATR_P = 20
STOP_ATR = 0.5
RR = 3.0
VERT = 180
MIN_GAP = 30
COST_R = 0.05
ADX_P, ADX_SLOPE = 14, 5
ST_PERIOD, ST_MULT = 10, 3.0           # canonical SuperTrend
BAR = "═" * 70
ROOT = None                            # set after clone → the FFM repo dir


def sh(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def step(n, title):
    print(f"\n{BAR}\n  STEP {n} — {title}\n{BAR}")


# ============================================================================
# STEP 0 — clone FFM + install
# ============================================================================
def bootstrap(workdir):
    global ROOT
    step(0, "Clone FFM library + install")
    if os.environ.get('CHRONOS_FT_CKPT'):
        print(f"  ⚠ CHRONOS_FT_CKPT is set — unset it for the vanilla backbone.")
    ROOT = Path(workdir).resolve() / 'Futures-Foundation-Model'
    if not ROOT.exists():
        sh(['git', 'clone', '--depth', '1', REPO_URL, str(ROOT)])
    else:
        print(f"  ✓ repo already present at {ROOT}")
    sh([sys.executable, '-m', 'pip', 'install', '-e', str(ROOT), '-q'])
    sh([sys.executable, '-m', 'pip', 'install', '-q',
        'xgboost', 'chronos-forecasting', 'torch'])
    sys.path.insert(0, str(ROOT))
    import importlib
    importlib.invalidate_caches()
    print(f"  ✓ FFM ready at {ROOT}")


# ============================================================================
# STEP 1 — copy CSVs + build feature parquet
# ============================================================================
def build_parquet(csv_dir, force):
    step(1, "Copy CSVs + build FFM feature parquet")
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    csvs = sorted(glob.glob(str(Path(csv_dir) / f'*_{TF}.csv')))
    if not csvs:
        raise SystemExit(f"  ✗ no *_{TF}.csv in {csv_dir}/")
    for f in csvs:
        shutil.copy(f, data_dir)
    print(f"  ✓ copied {len(csvs)} CSV(s) → {data_dir}")
    missing = [tk for tk in TICKERS if not (data_dir / f'{tk}_{TF}.csv').exists()]
    if missing:
        raise SystemExit(f"  ✗ missing {TF} CSVs for: {missing}")

    ffm_dir = ROOT / 'temp' / f'{TF}_FFM_Prepared_v2_causal'
    have = all((ffm_dir / f'{tk}_features.parquet').exists() for tk in TICKERS)
    if have and not force:
        print(f"  ✓ parquet already present — skipping (use --force-prep)")
        return ffm_dir
    from futures_foundation import prepare_data
    clean = ROOT / 'temp' / f'_raw_{TF}'                    # isolate this TF (glob safety)
    clean.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(clean / '*.csv')):
        os.remove(f)
    for tk in TICKERS:
        shutil.copy(data_dir / f'{tk}_{TF}.csv', clean)
    print(f"  building features (atr_period={ATR_P}) → {ffm_dir} ...")
    prepare_data(str(clean), str(ffm_dir), atr_period=ATR_P, force=True)
    print(f"  ✓ built {ffm_dir}/{{TICKER}}_features.parquet")
    return ffm_dir


# ============================================================================
# SuperTrend strategy (inlined; uses the cloned FFM library)
# ============================================================================
def make_labeler(ffm_dir):
    import numpy as np
    import pandas as pd
    from futures_foundation.pipeline._primitives import (
        apply_rr_barriers, compute_adx, compute_atr, compute_supertrend,
        max_favorable_rr)
    from futures_foundation.pipeline import evaluate as ev
    _META = {'_datetime', '_instrument', '_instrument_id',
             '_close', '_high', '_low', '_volume'}

    def _bars(tk):
        df = pd.read_csv(ROOT / 'data' / f'{tk}_{TF}.csv',
                         usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df[df['datetime'] >= df['datetime'].max()
                - pd.DateOffset(months=DATA_MONTHS)].reset_index(drop=True)
        ts = df['datetime'].to_numpy()
        o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
        atr = compute_atr(h, l, c, ATR_P)
        st_dir, _, _ = compute_supertrend(h, l, c, ST_PERIOD, ST_MULT)
        adx = compute_adx(h, l, c, ADX_P)
        adx_slope = np.full_like(adx, np.nan)
        adx_slope[ADX_SLOPE:] = adx[ADX_SLOPE:] - adx[:-ADX_SLOPE]
        ffm = pd.read_parquet(ffm_dir / f'{tk}_features.parquet')
        feat_cols = [x for x in ffm.columns if x not in _META]
        ffm['_datetime'] = pd.to_datetime(ffm['_datetime'], utc=True)
        ffm = ffm.set_index('_datetime')[feat_cols]
        aligned = ffm.reindex(df['datetime'], method='nearest',
                              tolerance=pd.Timedelta(seconds=30)).to_numpy(np.float32)
        return dict(ts=ts, o=o, h=h, l=l, c=c, atr=atr, lp=np.log(c),
                    st=st_dir, adx=adx, adx_slope=adx_slope,
                    ffm=aligned, feat_cols=feat_cols)

    class SuperTrendChronos:
        n_classes = 2

        def __init__(self):
            self._b = {tk: _bars(tk) for tk in TICKERS}
            self._feat_dim = len(next(iter(self._b.values()))['feat_cols']) + 2

        def calendar(self):
            return pd.concat(
                [pd.DataFrame({'item_id': tk, 'timestamp': B['ts'], 'target': B['c']})
                 for tk, B in self._b.items()], ignore_index=True)

        def _signals(self, B, lo, hi):
            ts, st, c = B['ts'], B['st'], B['c']
            n = len(c); last_l = last_s = -MIN_GAP
            for i in range(1, n - 1):
                if not (lo <= ts[i] < hi):
                    continue
                if i < CTX or i + 1 + VERT >= n:
                    continue
                if st[i] == st[i - 1]:
                    continue
                d = int(st[i])
                if d == 1 and i - last_l >= MIN_GAP:
                    last_l = i; yield i, 1
                elif d == -1 and i - last_s >= MIN_GAP:
                    last_s = i; yield i, -1

        def _barrier(self, B, i, d, tp_rr=None):
            a = B['atr'][i]
            if not np.isfinite(a) or a <= 0:
                return 'invalid', 0.0
            tp = float(tp_rr) if tp_rr is not None else RR
            entry = B['o'][i + 1]; risk = STOP_ATR * a
            sl = entry - risk if d == 1 else entry + risk
            res = apply_rr_barriers(B['h'], B['l'], B['c'], i + 1, d == 1, entry,
                                    sl, [tp], lookahead=VERT)[tp]
            return res['outcome'], res['realized_rr']

        def _max_rr(self, B, i, d):
            a = B['atr'][i]
            if not np.isfinite(a) or a <= 0:
                return 0.0
            entry = B['o'][i + 1]; risk = STOP_ATR * a
            sl = entry - risk if d == 1 else entry + risk
            return max_favorable_rr(B['h'], B['l'], i + 1, d == 1, entry, sl,
                                    lookahead=VERT)

        def build(self, lo, hi, test_start):
            C, Y, K = [], [], []
            for tk, B in self._b.items():
                ts, lp = B['ts'], B['lp']
                for i, d in self._signals(B, lo, hi):
                    if test_start is not None and ts[i + 1 + VERT] >= test_start:
                        continue
                    oc, _ = self._barrier(B, i, d)
                    if oc == 'invalid':
                        continue
                    mrr = self._max_rr(B, i, d)
                    C.append(lp[i - CTX + 1:i + 1])
                    Y.append(1 if oc == 'target_hit' else 0)
                    K.append((tk, i, d, float(mrr)))
            return C, np.array(Y), K

        def evaluate(self, keys, preds, risk_preds=None):
            R = []
            for idx, (key, p) in enumerate(zip(keys, preds)):
                if p != 1:
                    continue
                tk, i, d = key[0], key[1], key[2]
                tp_rr = None
                if risk_preds is not None:
                    tp_rr = float(np.clip(0.8 * float(risk_preds[idx]), 1.5, 8.0))
                oc, rr = self._barrier(self._b[tk], i, d, tp_rr=tp_rr)
                tp = tp_rr if tp_rr is not None else RR
                R.append(tp - COST_R if oc == 'target_hit'
                         else (-1.0 - COST_R if oc == 'stopped' else rr - COST_R))
            return np.array(R)

        def features(self, keys):
            out = np.zeros((len(keys), self._feat_dim), np.float32)
            g = lambda x: float(x) if np.isfinite(x) else 0.0
            for r, key in enumerate(keys):
                tk, i = key[0], key[1]
                B = self._b[tk]; ff = B['ffm'][i]
                out[r, :len(ff)] = ff
                out[r, -2] = g(B['adx'][i])
                out[r, -1] = g(B['adx_slope'][i])
            return np.nan_to_num(out, nan=np.nan, posinf=0.0, neginf=0.0)

    def audit():
        tk = TICKERS[0]; B = _bars(tk)
        h, l, c = B['h'], B['l'], B['c']; bad = 0
        for i in range(CTX, len(c), max(1, (len(c) - CTX) // 30)):
            st_c, _, _ = compute_supertrend(h[:i + 1], l[:i + 1], c[:i + 1],
                                            ST_PERIOD, ST_MULT)
            adx_c = compute_adx(h[:i + 1], l[:i + 1], c[:i + 1], ADX_P)
            for full, cau in ((B['st'][i], st_c[-1]), (B['adx'][i], adx_c[-1])):
                if np.isfinite(full) and abs(full - cau) > 1e-6:
                    bad += 1; break
        print(f"  causality ({tk}): {bad} mismatches "
              f"{'✅ CAUSAL' if bad == 0 else '❌ LOOKAHEAD'}")
        return bad == 0

    return SuperTrendChronos, audit, ev


# ============================================================================
# main
# ============================================================================
def main():
    global TF
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default='./csv',
                    help='dir with your {TICKER}_{tf}.csv files')
    ap.add_argument('--tf', default='3min',
                    help='bar timeframe (3min default, e.g. 1min/5min). Selects the '
                         'CSV suffix + feature parquet; same calc on the TF bars.')
    ap.add_argument('--workdir', default='.', help='where to clone FFM')
    ap.add_argument('--validate', action='store_true',
                    help='run the (slow) overfit-driven training loop before training')
    ap.add_argument('--no-train', action='store_true',
                    help='validation only — do NOT produce the joblib bundle')
    ap.add_argument('--force-produce', action='store_true',
                    help='build the bundle even if validation does not generalize')
    ap.add_argument('--skip-audit', action='store_true')
    ap.add_argument('--force-prep', action='store_true')
    ap.add_argument('--holdout-months', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-estimators', type=int, default=600)
    ap.add_argument('--max-depth', type=int, default=5)
    ap.add_argument('--output', default=None,
                    help='joblib output path (default: <FFM>/models/supertrend_chronos.joblib)')
    args = ap.parse_args()
    TF = args.tf

    bootstrap(args.workdir)
    ffm_dir = build_parquet(args.csv_dir, args.force_prep)
    Labeler, audit, ev = make_labeler(ffm_dir)
    cfg = (f"SuperTrend [{TF}] | {TICKERS} | period={ST_PERIOD}/mult={ST_MULT} | "
           f"SL={STOP_ATR}atr RR={RR} VERT={VERT}")

    if not args.skip_audit:
        step(2, "Causality audit")
        if not audit():
            raise SystemExit("  ✗ causality audit FAILED — aborting.")

    # Optional validation (slow) — the overfit-driven TRAINING LOOP.
    validated = None
    if args.validate:
        step("3a", "Validation — overfit-driven training loop")
        print(f"  {cfg}")
        validated = ev.run(Labeler(), loop=True, return_verdict=True)
        final = validated.get('final') or {}
        if final.get('all_pass'):
            print("\n  ✅ VALIDATION PASSED — generalizes on unseen data.")
        elif final.get('generalizes') is False:
            print("\n  ⚠️  DOES NOT GENERALIZE (overfit) — review before "
                  "shipping a bundle.")
        else:
            print("\n  ⚠️  Validation did not fully pass (edge below bar) — "
                  "review before shipping a bundle.")

    if args.no_train:
        print(f"\n{BAR}\n  DONE (validation only, --no-train)\n{BAR}")
        return

    # Gate: if validation ran and the model does not generalize, don't silently
    # ship a bundle that won't hold up — require an explicit override.
    if (validated and (validated.get('final') or {}).get('generalizes') is False
            and not args.force_produce):
        raise SystemExit("  ✗ ABORT: model did not generalize in validation. "
                         "Re-run with --force-produce to build the bundle anyway.")

    # PRODUCE — train the selection head + save the joblib (the output).
    step(3, "Produce — train selection head + save joblib bundle")
    from futures_foundation.pipeline.produce import train
    _name = ('supertrend_chronos.joblib' if TF == '3min'
             else f'supertrend_{TF}_chronos.joblib')
    out = args.output or str(ROOT / 'models' / _name)
    os.makedirs(Path(out).parent, exist_ok=True)
    lab = Labeler()
    print(f"  {cfg} | feat_dim={lab._feat_dim}+256 | "
          f"holdout={args.holdout_months}mo seed={args.seed} "
          f"n_est={args.n_estimators} depth={args.max_depth}")
    result = train(lab, holdout_months=args.holdout_months, seed=args.seed,
                   n_estimators=args.n_estimators, max_depth=args.max_depth,
                   output_path=out)
    print(f"\n{BAR}\n  ✅ DONE → bundle: {result['bundle_path']}  "
          f"({result['size_mb']:.1f} MB)\n{BAR}")


if __name__ == '__main__':
    main()

