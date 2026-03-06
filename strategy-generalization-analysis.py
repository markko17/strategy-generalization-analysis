"""
Strategy Generalization Analysis
=================================
Analyzes how well walk-forward-optimized strategies generalize from history
(in-sample + out-of-sample windows) into unseen "live proxy" windows.

IMPORTANT: Strategy files analyzed by this script must be produced by
`run_strategies.py` (or the same backtesting framework). Each strategy
subfolder under `root_dir` must contain a `.txt` results file with lines
in the format:

    W01 IS   PF: 1.23  ROI: $456  Trades: 80  Win: 52.3%
    W01 OOS  PF: 1.10  ROI: $210  Trades: 40  Win: 54.0%
    W01 IS+ENT  PF: 1.18  ...         <- robustness variant (optional)
    W01 OOS+ENT PF: 1.05  ...
    ...

These files are generated automatically when you run run_strategies.py with
the following robustness switches enabled in BACKTESTER_OVERRIDES:
    ENTRY_DRIFT       = True   -> produces IS+ENT / OOS+ENT lines
    INDICATOR_VARIANCE= True   -> produces IS+IND / OOS+IND lines
    FEE_SHOCK         = True   -> produces IS+FEE / OOS+FEE lines
    SLIPPAGE_SHOCK    = True   -> produces IS+SLI / OOS+SLI lines

You must enable at least one robustness switch to use any test with
"Eligible for manual export selection (robustness-required): YES".

WORKFLOW OVERVIEW:
1. Run run_strategies.py to generate strategy result folders under your output directory.
2. Set `root_dir` below to that same output directory.
3. Run this script once to see the "TEST ID MAPPING" output.
4. Pick a test ID marked as "Eligible ... : YES" and set SELECT_TEST_NUMBER.
5. Re-run to export an Excel file with passing strategies and portfolio stats.

QUICK START:
    # Step 1: Point to your strategies folder
    root_dir = r"C:\path\to\your\strategy_output_folder"

    # Step 2: Run and read the TEST ID MAPPING printed to console
    # Step 3: Choose a test and set SELECT_TEST_NUMBER (e.g. 10)
    # Step 4: Re-run -> Excel exported to root_dir

See the CONFIG section below for all tunable parameters.
"""

import os
import re
import math
import sys
import subprocess
import json
import tempfile
from collections import defaultdict
from statistics import median
from builtins import print as _builtin_print
import numpy as np
import matplotlib.pyplot as plt

# Excel export
import pandas as pd

# Beta-Binomial lower bound
try:
    from scipy.stats import beta as beta_dist
except ImportError as e:
    raise ImportError("scipy is required for beta.ppf (pip install scipy).") from e

# ---------------------------------------------------------------------------
# USER CONFIGURATION: Set this to the folder produced by run_strategies.py
# ---------------------------------------------------------------------------
# This should be the `base_output` folder you passed to run_strategies.py.
# Example (Windows):  r"C:\Strategies\MyRun"
# Example (Linux/Mac): "/home/user/strategies/myrun"
root_dir = r"YOUR_STRATEGY_OUTPUT_FOLDER_HERE"

RE_WINDOW_LINE_DETECT = re.compile(r"^\s*W(\d{2,3})\s+(IS|OOS)\b", re.IGNORECASE)

def log_line(*args, **kwargs):
    """Single output wrapper used across the script for consistent messaging."""
    _builtin_print(*args, **kwargs)

def detect_windows_in_file(root_dir_path: str, preferred_windows: int = None):
    """
    Detect total window count from strategy .txt files under root_dir.
    If preferred_windows is provided, prefer the first file that matches it.
    Returns (windows_in_file, file_used).
    """
    first_txt = None
    first_txt_windows = None
    for subdir, _, files in os.walk(root_dir_path):
        for fn in files:
            if fn.lower().endswith(".txt"):
                candidate = os.path.join(subdir, fn)
                max_w = 0
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        m = RE_WINDOW_LINE_DETECT.match(line)
                        if not m:
                            continue
                        w = int(m.group(1))
                        if w > max_w:
                            max_w = w
                if max_w <= 0:
                    continue
                if first_txt is None:
                    first_txt = candidate
                    first_txt_windows = max_w
                if preferred_windows is not None and max_w == preferred_windows:
                    return max_w, candidate

    if not first_txt:
        raise FileNotFoundError(f"No .txt strategy files found under root_dir: {root_dir_path}")
    return first_txt_windows, first_txt

# ---------------------------------------------------------------------------
# CONFIG - Tune these settings before running
# ---------------------------------------------------------------------------
# Core window geometry for each run:
# - EXPECTED_WINDOWS defines how many windows are analyzed in one active run.
# - WINDOWS_IN_FILE is either auto-detected or manually set and is the total windows in source files.
# How many WFO windows exist in each run_strategies.py output file.
# Must match the number of windows your backtester produced.
# Typical values: 4, 6, 7. Must be >= 2.
EXPECTED_WINDOWS = 6
AUTO_DETECT_WINDOWS_IN_FILE = True
WINDOWS_IN_FILE = None  # set manual value only when AUTO_DETECT_WINDOWS_IN_FILE=False
PF_OK = 1.0
ENABLE_PLOTS = True          # disable for faster/headless runs

# Execution modes:
# - LIVE_MODE: shifts the active window block forward by 2 windows and treats
#   the most recent windows as current (not as held-out live proxies).
#   Use this when deploying live and you want to re-apply the filter without
#   a held-out test period.
# - META_MODE: re-runs the script across every possible sliding window block
#   and prints a summary table, showing temporal stability across all offsets.
#   Set to False for a single-run analysis.
LIVE_MODE = False
META_MODE = False

# Portfolio generation controls:
# - PORTFOLIO_SIZE: number of strategies per sampled portfolio (e.g. 5 or 20).
# - MAX_PORTFOLIO_COMBOS: maximum number of portfolios to sample. Increase for
#   more statistical precision, decrease for faster runs.
# - TOP_STRATEGIES_POOL: portfolios are sampled from the top-N strategies
#   ranked by TOP_STRATEGY_RANK_METRIC ("sharpe", "pf", or "maxdd").
PORTFOLIO_SIZE = 20
MAX_PORTFOLIO_COMBOS = 10000
BUILD_PAIR_PORTFOLIOS = False  # compatibility flag; pair generation is not used
TOP_STRATEGIES_POOL = 24       # random portfolios are sampled from top-N history performers
TOP_STRATEGY_RANK_METRIC = "pf"  # accepted values: "sharpe", "pf", "maxdd"

# ---------------------------------------------------------------------------
# RISK & MILESTONE THRESHOLDS (used by portfolio challenge-style tests)
# ---------------------------------------------------------------------------
# These simulate prop-firm-style challenges on the live proxy windows:
# - TARGET_R / DRAWDOWN_LIMIT_R / DAY_LOSS_LIMIT_R: R-unit based thresholds.
# - USE_PCT_MODE: if True, converts everything to % of ACCOUNT_BALANCE.
# - RISK_PCT_PER_R: % of equity risked per 1R trade (compounded).
TARGET_R = 10          # e.g., reach 5R profit
DRAWDOWN_LIMIT_R = -10 # e.g., avoid losing 3R before target
DAY_LOSS_LIMIT_R = -5 # e.g., avoid losing 4R in a single day before target
USE_PCT_MODE = True  # if True, use percentage-based equity milestones instead of R
ACCOUNT_BALANCE = 1000.0  # starting balance for percentage mode
RISK_PCT_PER_R = 0.1      # e.g., 0.01 = 1% of equity per 1R; compounded per trade
R_UNIT_SCALE = 10          # fixed-risk mode: scale Live_R by this factor (e.g., 0.2 means 1R=0.2 units)

# Backtest-only mode:
# - When Backtest_only=True, the script skips the full pipeline and only runs
#   diagnostic plots/statistics for a specific window block, then exits.
# - Useful for inspecting distribution curves and equity curves without
#   running a full analysis.
# - The window selector below defines the split between history and live windows.
Backtest_only = False
Backtest_only_plot_selected_portfolio = False
Backtest_all_strategies_distribution_curve = True
Backtest_all_portfolios_distribution_curve = True
# Backtest-only window block selector:
# start=1 -> history W01..W04, live W05..W06
# start=3 -> history W03..W06, live W07..W08
Backtest_only_window_start = 1
Backtest_only_history_windows = 4
Backtest_only_live_windows = 2
backtest_list = {
    "ATR_x_EMA100_normalized_price_src_skew_SL3": 1.0,
    "SMA_x_SMA100_roc_BW_filter_SL2": 1.0,
    "SMA_x_SMA50_fold_dev_InsideBar_SL2": 1.0,
    "RSI_x_EMA50_bias_calc_skew_SL3": 1.0,
    "ATR_x_EMA100_normalized_price_src_SL3": 1.0,
    "RSI_x_EMA20_skew_SL3": 1.0,
    "EMA_x_EMA50_roc_atr_pct0.8_SL3": 1.0,
    "RSI_x_EMA50_atr_pct_SL3": 1.0,
    "ATR_x_EMA100_normalized_price_src_RSIge40_SL3": 1.0,
    "RSI_x_EMA100_accel_src_BW_filter_SL3": 1.0,
    "RSI_x_EMA100_normalized_price_src_atr_pct_SL3": 1.0,
    "ATR_x_EMA200_accel_src_Pge0.7_SL3": 1.0,
    "MACD(42,110)_roc_InsideBar_SL3": 1.0,
    "STOCHK_SMA_55_normalized_price_src_Pge0.7_SL2": 1.0,
    "STOCHK_EMA_89_fold_dev_calc_Pge0.7_SL3": 1.0,
    "ATR_x_EMA200_slope_src_SL3": 1.0,
    "ATR_x_EMA100_rank_resid_src_skew_SL3": 1.0,
    "RSI_x_SMA50_slope_calc_Pge0.8_SL2": 1.0,
    "PPO_x_EMA50_volZ_calc_Pge0.7_SL2": 1.0,
    "ATR_x_EMA100_volZ_src_Pge0.8_SL3": 1.0,
}

# ---------------------------------------------------------------------------
# TEST SELECTION
# ---------------------------------------------------------------------------
# STEP 1: Run the script once with any SELECT_TEST_NUMBER.
#         Read the "TEST ID MAPPING" block printed to the console.
# STEP 2: Choose a test ID where:
#         "Eligible for manual export selection (robustness-required): YES"
#         These tests require that your strategy files contain robustness
#         lines (IS+ENT, OOS+ENT, etc.) produced by run_strategies.py when
#         ENTRY_DRIFT / INDICATOR_VARIANCE / FEE_SHOCK etc. are enabled.
# STEP 3: Set SELECT_TEST_NUMBER to that number and re-run to export.
#
# DO NOT pick base-only quality-gate tests (e.g. ">100 trades + dedup")
# for the main export — those have no robustness requirement.
SELECT_TEST_NUMBER = 10

TEST_ID_MAP_TAGS = ["ENT", "IND", "FEE", "SLI"]

def _startup_build_pipeline_names(tags: list = None) -> list:
    tags = TEST_ID_MAP_TAGS if tags is None else tags
    names = ["Strategies with >100 trades total (base IS+OOS across all windows) + deduped"]
    for tag in tags:
        names.append(f"Strategies with 2 or more profitable windows (base) + IS+{tag} + OOS")
        names.append(f"Strategies with 2 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")
        names.append(f"Strategies with 3 or more profitable windows (base) + IS+{tag} + OOS")
        names.append(f"Strategies with 3 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")
        if EXPECTED_WINDOWS >= 4:
            names.append(f"Strategies with 4 profitable windows (base) + IS+{tag} + OOS")
            names.append(f"Strategies with 4 profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")
    return names

def _print_startup_test_id_mapping_first():
    """
    Prints test ID mapping immediately on startup, before any other runtime work.
    """
    pipeline_names = _startup_build_pipeline_names()

    log_line("\n==================== TEST ID MAPPING (startup-first) ====================\n")
    for idx, pname in enumerate(pipeline_names, start=1):
        log_line(f"test {idx}: {pname}")
    log_line("")

_print_startup_test_id_mapping_first()

if AUTO_DETECT_WINDOWS_IN_FILE:
    WINDOWS_IN_FILE, WINDOWS_DETECTED_FROM = detect_windows_in_file(root_dir, preferred_windows=EXPECTED_WINDOWS)
else:
    if WINDOWS_IN_FILE is None:
        WINDOWS_IN_FILE = EXPECTED_WINDOWS
    WINDOWS_DETECTED_FROM = "manual"

if WINDOWS_IN_FILE < EXPECTED_WINDOWS:
    raise ValueError("WINDOWS_IN_FILE must be >= EXPECTED_WINDOWS.")

ENV_WINDOW_OFFSET_OVERRIDE = "WFO_WINDOW_OFFSET_OVERRIDE"
ENV_META_CHILD = "WFO_META_CHILD"
ENV_META_PLOT_DIR = "WFO_META_PLOT_DIR"
ENV_META_PLOT_PREFIX = "WFO_META_PLOT_PREFIX"
ENV_META_DEFER_PLOTS = "WFO_META_DEFER_PLOTS"
ENV_FORCE_FULL_PIPELINE = "WFO_FORCE_FULL_PIPELINE"
ENV_BACKTEST_PORT_ROI_OUT = "WFO_BACKTEST_PORTFOLIO_ROI_OUT"
ENV_BACKTEST_SKIP_XLSX = "WFO_BACKTEST_SKIP_XLSX"

if os.environ.get(ENV_FORCE_FULL_PIPELINE) == "1":
    Backtest_only = False

_window_offset_override_raw = os.environ.get(ENV_WINDOW_OFFSET_OVERRIDE, "").strip()
if _window_offset_override_raw:
    try:
        WINDOW_OFFSET = int(_window_offset_override_raw)
    except ValueError as e:
        raise ValueError(
            f"{ENV_WINDOW_OFFSET_OVERRIDE} must be an integer, got: {_window_offset_override_raw!r}"
        ) from e
else:
    WINDOW_OFFSET = max(0, WINDOWS_IN_FILE - EXPECTED_WINDOWS)

_max_window_offset = WINDOWS_IN_FILE - EXPECTED_WINDOWS
if WINDOW_OFFSET < 0 or WINDOW_OFFSET > _max_window_offset:
    raise ValueError(
        f"WINDOW_OFFSET={WINDOW_OFFSET} is invalid for WINDOWS_IN_FILE={WINDOWS_IN_FILE} "
        f"and EXPECTED_WINDOWS={EXPECTED_WINDOWS}. Allowed range: 0..{_max_window_offset}."
    )

WINDOW_FILE_START = 1 + WINDOW_OFFSET
WINDOW_FILE_END = WINDOW_FILE_START + EXPECTED_WINDOWS - 1

# Live proxy windows:
# - If 7 windows: W06 & W07
# - If 4 windows: W03 & W04
# - Otherwise: last two windows
if EXPECTED_WINDOWS == 7:
    LIVE_WA = 6
    LIVE_WB = 7
elif EXPECTED_WINDOWS == 4:
    LIVE_WA = 3
    LIVE_WB = 4
else:
    LIVE_WA = EXPECTED_WINDOWS - 1
    LIVE_WB = EXPECTED_WINDOWS

# Indices in Python lists
LIVE_IA = LIVE_WA - 1
LIVE_IB = LIVE_WB - 1

# Windows used for "history" (W1..W5 in 7-window case; W1..W2 in 4-window case)
HIST_WINDOWS = max(1, EXPECTED_WINDOWS - 2)   # always excludes the last 2 proxy windows
HIST_IDXS = list(range(HIST_WINDOWS))         # 0..HIST_WINDOWS-1
HIST_W_NUMS = list(range(1, HIST_WINDOWS + 1))

# Live-mode re-application skips the earliest windows and reuses the most recent ones
LIVE_MODE_SKIP = 2
LIVE_MODE_START = min(EXPECTED_WINDOWS, 1 + LIVE_MODE_SKIP)
LIVE_MODE_WINDOWS = list(range(LIVE_MODE_START, EXPECTED_WINDOWS + 1)) if LIVE_MODE_START <= EXPECTED_WINDOWS else []

# Base tiers (keep your original tier names / concept, but "RB" is now per TAG)
TIERS_BASE = ["IS"]
TIERS_RB = ["IS_ISRB", "IS_ISRB_OOS", "IS_ISRB_OOS_OOSRB"]

def log_runtime_configuration():
    """Emit a readable summary of runtime controls and derived window roles."""
    log_line("\n==================== RUNTIME CONFIGURATION ====================\n")
    log_line(f"Root directory: {root_dir}")
    log_line(
        "Window policy: "
        f"EXPECTED_WINDOWS={EXPECTED_WINDOWS}, WINDOWS_IN_FILE={WINDOWS_IN_FILE}, "
        f"active file windows=W{WINDOW_FILE_START:02d}..W{WINDOW_FILE_END:02d}, offset={WINDOW_OFFSET}"
    )
    log_line(
        "Role mapping: "
        f"history windows=W01..W{HIST_WINDOWS:02d}, live proxies=W{LIVE_WA:02d}&W{LIVE_WB:02d}"
    )
    log_line(
        "Mode flags: "
        f"LIVE_MODE={LIVE_MODE}, META_MODE={META_MODE}, Backtest_only={Backtest_only}, ENABLE_PLOTS={ENABLE_PLOTS}"
    )
    log_line(
        "Portfolio config: "
        f"size={PORTFOLIO_SIZE}, max_combos={MAX_PORTFOLIO_COMBOS}, top_rank_metric={TOP_STRATEGY_RANK_METRIC}, "
        f"top_pool={TOP_STRATEGIES_POOL}"
    )
    if USE_PCT_MODE:
        log_line(
            "Risk model: percentage-compounding "
            f"(ACCOUNT_BALANCE={ACCOUNT_BALANCE}, RISK_PCT_PER_R={RISK_PCT_PER_R})"
        )
    else:
        log_line(f"Risk model: fixed-risk R units (R_UNIT_SCALE={R_UNIT_SCALE})")
    if Backtest_only:
        log_line(
            "Backtest-only windows: "
            f"start=W{Backtest_only_window_start:02d}, history_count={Backtest_only_history_windows}, "
            f"live_count={Backtest_only_live_windows}"
        )

def _extract_meta_run_summary(stdout_text: str) -> dict:
    def _find_int(pattern: str):
        m = re.search(pattern, stdout_text, flags=re.MULTILINE)
        return int(m.group(1)) if m else None

    def _find_float(pattern: str):
        m = re.search(pattern, stdout_text, flags=re.MULTILINE)
        return float(m.group(1)) if m else None

    return {
        "exported_portfolios": _find_int(r"^Portfolios \(\d+-strategy equal-weight\):\s*(\d+)\s*$"),
        "total_portfolios": _find_int(r"^Total portfolios:\s*(\d+)\s*$"),
        "profitable": _find_int(r"^Profitable .*?:\s*(\d+)\s*$"),
        "unprofitable": _find_int(r"^Unprofitable .*?:\s*(\d+)\s*$"),
        "avg_live_pct": _find_float(r"^Average W\d+&W\d+ result \(%\):\s*([+-]?\d+(?:\.\d+)?)%\s*$"),
        "avg_maxdd_pct": _find_float(r"^Average max drawdown \(%\):\s*([+-]?\d+(?:\.\d+)?)%\s*$"),
        "avg_prof_dd_r": _find_float(r"^Avg max DD \(profitable, R\):\s*([+-]?\d+(?:\.\d+)?)\s*$"),
        "avg_unprof_dd_r": _find_float(r"^Avg max DD \(unprofitable, R\):\s*([+-]?\d+(?:\.\d+)?)\s*$"),
    }

def _maybe_run_meta_mode():
    if not META_MODE:
        return
    if os.environ.get(ENV_META_CHILD) == "1":
        return

    max_offset = WINDOWS_IN_FILE - EXPECTED_WINDOWS
    if max_offset < 2:
        log_line(
            f"[META MODE] Disabled for this run: requires at least EXPECTED_WINDOWS + 2 windows "
            f"(>= {EXPECTED_WINDOWS + 2}), but detected {WINDOWS_IN_FILE}."
        )
        return

    script_path = os.path.abspath(__file__) if "__file__" in globals() else None
    if not script_path or not os.path.isfile(script_path):
        log_line("[META MODE] Could not resolve script path; running normal mode instead.")
        return

    run_offsets = list(range(0, max_offset + 1))
    total_runs = len(run_offsets)
    meta_results = []
    meta_plot_dir = tempfile.mkdtemp(prefix="wfo_meta_plots_") if ENABLE_PLOTS else None

    log_line("\n==================== META MODE ====================\n")
    log_line(f"Expected windows per run: {EXPECTED_WINDOWS}")
    log_line(f"Detected windows in file: {WINDOWS_IN_FILE}")
    log_line(f"Sliding runs to execute: {total_runs} (offsets 0..{max_offset})")
    log_line(f"Selected test number (reused each run): {SELECT_TEST_NUMBER}")

    for run_idx, offset in enumerate(run_offsets, start=1):
        file_start = 1 + offset
        file_end = file_start + EXPECTED_WINDOWS - 1
        live_a_file = file_start + (LIVE_WA - 1)
        live_b_file = file_start + (LIVE_WB - 1)

        log_line("\n" + "=" * 60)
        log_line(
            f"[META RUN {run_idx}/{total_runs}] File windows W{file_start:02d}-W{file_end:02d} | "
            f"Live proxies W{live_a_file:02d}-W{live_b_file:02d} | offset={offset}"
        )
        log_line("=" * 60)

        env = os.environ.copy()
        env[ENV_WINDOW_OFFSET_OVERRIDE] = str(offset)
        env[ENV_META_CHILD] = "1"
        if ENABLE_PLOTS and meta_plot_dir:
            env[ENV_META_DEFER_PLOTS] = "1"
            env[ENV_META_PLOT_DIR] = meta_plot_dir
            env[ENV_META_PLOT_PREFIX] = f"run{run_idx:02d}_W{file_start:02d}_W{file_end:02d}"
        env.setdefault("PYTHONIOENCODING", "utf-8")

        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path) or None,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
        )

        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        if stdout_text:
            log_line(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
        if stderr_text.strip():
            log_line("\n[META RUN STDERR]")
            log_line(stderr_text, end="" if stderr_text.endswith("\n") else "\n")

        run_summary = _extract_meta_run_summary(stdout_text)
        run_summary.update({
            "run_idx": run_idx,
            "offset": offset,
            "file_start": file_start,
            "file_end": file_end,
            "live_a_file": live_a_file,
            "live_b_file": live_b_file,
            "returncode": proc.returncode,
        })
        meta_results.append(run_summary)

        if proc.returncode != 0:
            log_line(f"\n[META MODE] Stopping because run {run_idx}/{total_runs} failed with exit code {proc.returncode}.")
            raise SystemExit(proc.returncode)

    log_line("\n==================== META SUMMARY ====================\n")
    for r in meta_results:
        total_pf = r.get("total_portfolios")
        exported_pf = r.get("exported_portfolios")
        profitable = r.get("profitable")
        unprofitable = r.get("unprofitable")
        avg_live_pct = r.get("avg_live_pct")
        avg_maxdd_pct = r.get("avg_maxdd_pct")
        avg_prof_dd_r = r.get("avg_prof_dd_r")
        avg_unprof_dd_r = r.get("avg_unprof_dd_r")

        log_line(
            f"Run {r['run_idx']}/{total_runs} | W{r['file_start']:02d}-W{r['file_end']:02d} "
            f"(live W{r['live_a_file']:02d}-W{r['live_b_file']:02d})"
        )
        log_line(
            f"  Total portfolios: {total_pf if total_pf is not None else 'n/a'} | "
            f"Exported portfolios: {exported_pf if exported_pf is not None else 'n/a'} | "
            f"Profitable: {profitable if profitable is not None else 'n/a'} | "
            f"Unprofitable: {unprofitable if unprofitable is not None else 'n/a'}"
        )
        if avg_live_pct is not None or avg_maxdd_pct is not None:
            log_line(f"  Avg live result (%): {avg_live_pct:.2f}" if avg_live_pct is not None else "  Avg live result (%): n/a")
            log_line(f"  Avg max drawdown (%): {avg_maxdd_pct:.2f}" if avg_maxdd_pct is not None else "  Avg max drawdown (%): n/a")
        if avg_prof_dd_r is not None or avg_unprof_dd_r is not None:
            log_line(f"  Avg max DD (profitable, R): {avg_prof_dd_r:.2f}" if avg_prof_dd_r is not None else "  Avg max DD (profitable, R): n/a")
            log_line(f"  Avg max DD (unprofitable, R): {avg_unprof_dd_r:.2f}" if avg_unprof_dd_r is not None else "  Avg max DD (unprofitable, R): n/a")

    if ENABLE_PLOTS and meta_plot_dir and os.path.isdir(meta_plot_dir):
        saved_plot_files = sorted(
            [os.path.join(meta_plot_dir, fn) for fn in os.listdir(meta_plot_dir) if fn.lower().endswith(".png")]
        )
        if saved_plot_files:
            log_line("\n==================== META PLOTS (DEFERRED) ====================\n")
            log_line(f"Opening {len(saved_plot_files)} matplotlib figure(s) after all meta runs...")
            for img_path in saved_plot_files:
                try:
                    img = plt.imread(img_path)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.set_title(os.path.basename(img_path))
                    ax.axis("off")
                except Exception as e:
                    log_line(f"[META MODE] Failed to open deferred plot {img_path}: {e}")
            plt.show()
        else:
            log_line("\n[META MODE] No deferred plots were captured.\n")

    raise SystemExit(0)

if os.environ.get(ENV_META_CHILD) != "1":
    log_runtime_configuration()

_maybe_run_meta_mode()

# ---------------------------------------------------------------------------
# TEST ID INFRASTRUCTURE + EARLY MAPPING PRINT
# ---------------------------------------------------------------------------
# Moved before the file walk so the mapping is printed on startup,
# even if root_dir doesn't exist or contains no files yet.

# --- compact pipeline IDs ("test 1", "test 2", ...)
pipeline_rows = []  # every row = (test_id, metric, lower_bound, point_est, n)

test_id_map = {}          # full_pipeline_name -> "test N"
test_id_reverse = {}      # "test N" -> full_pipeline_name
_next_test_id = 1

def get_test_id(full_name: str) -> str:
    global _next_test_id
    full_name = full_name.strip()
    if full_name in test_id_map:
        return test_id_map[full_name]
    tid = f"test {_next_test_id}"
    _next_test_id += 1
    test_id_map[full_name] = tid
    test_id_reverse[tid] = full_name
    return tid

def _add_pipeline_row(pipeline_name: str, metric: str, n: int, point_est_pct: float):
    lb = beta_lower_bound_95(point_est_pct, n)
    tid = get_test_id(pipeline_name)
    pipeline_rows.append({
        "pipeline": tid,
        "metric": metric,
        "lower_bound": lb,
        "point_est": point_est_pct,
        "n": n
    })

def _print_block_and_collect(pipeline_name: str, bucket: dict):
    n = int(bucket["count"])
    wa_pct = _pct(bucket["wa"], n)
    wb_pct = _pct(bucket["wb"], n)
    wab_pct = _pct(bucket["wab"], n)

    log_line(f"\n{pipeline_name}")
    log_line(f"  - Matched strategies: {n}")
    log_line(f"  - Live proxy W{LIVE_WA} profitable rate: {wa_pct:.2f}%")
    log_line(f"  - Live proxy W{LIVE_WB} profitable rate: {wb_pct:.2f}%")
    log_line(f"  - Both live proxies profitable rate: {wab_pct:.2f}%")

    _add_pipeline_row(pipeline_name, f"W{LIVE_WA}", n, wa_pct)
    _add_pipeline_row(pipeline_name, f"W{LIVE_WB}", n, wb_pct)
    _add_pipeline_row(pipeline_name, f"W{LIVE_WA}&W{LIVE_WB}", n, wab_pct)

def _agg_rb_bucket(tag, tier, k_min=None, k_exact=None):
    out = {"count": 0, "wa": 0, "wb": 0, "wab": 0}
    if tag not in stats_by_rb:
        return out

    if k_exact is not None:
        ks = [k_exact]
    else:
        k_min = 1 if k_min is None else k_min
        ks = list(range(k_min, EXPECTED_WINDOWS + 1))

    for k in ks:
        s = stats_by_rb[tag][k][tier]
        out["count"] += s["count"]
        out["wa"] += s["wa"]
        out["wb"] += s["wb"]
        out["wab"] += s["wab"]
    return out

# -------------------------------
# TEST ID MAPPING (kept so you can choose SELECT_TEST_NUMBER)
# -------------------------------
# HOW TO READ THIS BLOCK:
# - Every "test N" is one pipeline/filter definition.
# - "Requirements" tells which robustness runs must exist and pass for a strategy.
# - If you require robustness-tested strategies, only choose tests marked:
#   "Eligible for manual export selection (robustness-required): YES"
def _describe_pipeline_human_readable(pipeline_name: str) -> str:
    """
    Convert internal pipeline labels into a short explanation of what the test filters for.
    This helps when selecting SELECT_TEST_NUMBER.
    """
    s = pipeline_name.strip()
    if s.startswith("Strategies with >100 trades total"):
        return "Base quality gate: at least 100 total trades across windows + deduplicated signatures."

    m = re.match(
        r"^Strategies with (\d+) or more profitable windows \(base\) \+ IS\+([A-Z+]+) \+ OOS( \+ OOS\+\2)?$",
        s
    )
    if m:
        kmin = int(m.group(1))
        tag = m.group(2).upper()
        with_oos_rb = (m.group(3) is not None)
        if with_oos_rb:
            return (
                f"Robustness funnel: base has >= {kmin} profitable IS windows, "
                f"IS+{tag} must pass, OOS must pass, and OOS+{tag} must also pass."
            )
        return (
            f"Robustness funnel: base has >= {kmin} profitable IS windows, "
            f"then IS+{tag} and OOS must pass."
        )

    m = re.match(
        r"^Strategies with (\d+) profitable windows \(base\) \+ IS\+([A-Z+]+) \+ OOS( \+ OOS\+\2)?$",
        s
    )
    if m:
        kexact = int(m.group(1))
        tag = m.group(2).upper()
        with_oos_rb = (m.group(3) is not None)
        if with_oos_rb:
            return (
                f"Robustness funnel (exact-k): base has exactly {kexact} profitable IS windows, "
                f"then IS+{tag}, OOS, and OOS+{tag} must pass."
            )
        return (
            f"Robustness funnel (exact-k): base has exactly {kexact} profitable IS windows, "
            f"then IS+{tag} and OOS must pass."
        )
    return "Pipeline meaning could not be auto-parsed from its label."

def parse_pipeline_descriptor(name: str) -> dict:
    """Parse pipeline text into a structured descriptor for selection and requirement checks."""
    s = name.strip()
    if s.startswith("Strategies with >100 trades total"):
        return {"type": "GT100_DEDUP"}

    m = re.match(
        r"^Strategies with (\d+) or more profitable windows \(base\) \+ IS\+([A-Z+]+) \+ OOS( \+ OOS\+\2)?$",
        s
    )
    if m:
        kmin = int(m.group(1))
        tag = m.group(2).upper()
        with_oos_rb = (m.group(3) is not None)
        return {"type": "RB_FUNNEL", "k_mode": "MIN", "k": kmin, "tag": tag, "with_oos_rb": with_oos_rb}

    m = re.match(
        r"^Strategies with (\d+) profitable windows \(base\) \+ IS\+([A-Z+]+) \+ OOS( \+ OOS\+\2)?$",
        s
    )
    if m:
        kexact = int(m.group(1))
        tag = m.group(2).upper()
        with_oos_rb = (m.group(3) is not None)
        return {"type": "RB_FUNNEL", "k_mode": "EXACT", "k": kexact, "tag": tag, "with_oos_rb": with_oos_rb}

    return {"type": "UNKNOWN", "raw": s}

def _pipeline_requirements_human_readable(pipeline_name: str) -> str:
    """
    Describe required line availability in each strategy file for this test.
    This is the key section for test selection.
    """
    desc = parse_pipeline_descriptor(pipeline_name)
    if desc["type"] == "RB_FUNNEL":
        k_part = f">= {desc['k']}" if desc["k_mode"] == "MIN" else f"exactly {desc['k']}"
        req = (
            f"Requires robustness runs: base IS has {k_part} profitable windows, "
            f"plus IS+{desc['tag']} and base OOS lines must exist and pass."
        )
        if desc.get("with_oos_rb"):
            req += f" Also requires OOS+{desc['tag']} lines to exist and pass."
        return req
    if desc["type"] == "GT100_DEDUP":
        return "No robustness-tag requirement (base-only quality gate)."
    return "Requirements could not be parsed from pipeline label."

def _pre_print_test_id_map():
    """
    Pre-populate test_id_map / test_id_reverse and print the TEST ID MAPPING
    before any file walking begins. Uses the same pipeline registration order
    as the main analysis so SELECT_TEST_NUMBER values remain consistent.
    """
    # Register pipelines in same order the main analysis would
    get_test_id("Strategies with >100 trades total (base IS+OOS across all windows) + deduped")
    for tag in TEST_ID_MAP_TAGS:
        get_test_id(f"\nStrategies with 2 or more profitable windows (base) + IS+{tag} + OOS")
        get_test_id(f"Strategies with 2 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")
        get_test_id(f"\nStrategies with 3 or more profitable windows (base) + IS+{tag} + OOS")
        get_test_id(f"Strategies with 3 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")
        if EXPECTED_WINDOWS >= 4:
            get_test_id(f"\nStrategies with 4 profitable windows (base) + IS+{tag} + OOS")
            get_test_id(f"Strategies with 4 profitable windows (base) + IS+{tag} + OOS + OOS+{tag}")

    log_line("\n==================== TEST ID MAPPING (preliminary) ====================\n")
    log_line("Static requirements-only mapping (independent of current files). A complete version is printed after the full file walk.")
    log_line("Test selection rule: only robustness-funnel tests (IS+TAG / OOS+TAG pipelines) are eligible for export.\n")

    for tid in sorted(test_id_reverse.keys(), key=lambda x: int(x.split()[1])):
        pname = test_id_reverse[tid]
        pdesc = parse_pipeline_descriptor(pname)
        eligible = "YES" if pdesc["type"] == "RB_FUNNEL" else "NO"
        log_line(f"{tid}: {pname.strip()}")
        log_line(f"    Meaning: {_describe_pipeline_human_readable(pname)}")
        log_line(f"    Requirements: {_pipeline_requirements_human_readable(pname)}")
        log_line(f"    Eligible for manual export selection (robustness-required): {eligible}")


if os.environ.get(ENV_META_CHILD) != "1":
    _pre_print_test_id_map()

_ORIG_PLT_SHOW = plt.show
_meta_show_batch_counter = 0

def _wfo_show_wrapper(*args, **kwargs):
    global _meta_show_batch_counter
    if os.environ.get(ENV_META_DEFER_PLOTS) != "1":
        return _ORIG_PLT_SHOW(*args, **kwargs)

    plot_dir = os.environ.get(ENV_META_PLOT_DIR, "").strip()
    prefix = os.environ.get(ENV_META_PLOT_PREFIX, "meta")
    if not plot_dir:
        return None

    os.makedirs(plot_dir, exist_ok=True)
    fig_nums = list(plt.get_fignums())
    if not fig_nums:
        return None

    _meta_show_batch_counter += 1
    batch_id = _meta_show_batch_counter
    for seq, fig_num in enumerate(fig_nums, start=1):
        fig = plt.figure(fig_num)
        out_name = f"{prefix}_show{batch_id:02d}_fig{seq:02d}.png"
        out_path = os.path.join(plot_dir, out_name)
        try:
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
        except Exception:
            # Fallback without bbox if a backend/artist complains
            fig.savefig(out_path, dpi=120)
        plt.close(fig)
    return None

plt.show = _wfo_show_wrapper

# -------------------------------
# STATS STRUCTURES
# -------------------------------
# Baseline stats: group by k=number of profitable IS windows, evaluate LIVE using base OOS
stats_base = {
    k: {
        tier: {"count": 0, "wa": 0, "wb": 0, "wab": 0}
        for tier in TIERS_BASE
    }
    for k in range(1, EXPECTED_WINDOWS + 1)
}

# Per-robustness stats: tag -> k -> tier -> counts
stats_by_rb = {}  # filled dynamically: stats_by_rb[tag][k][tier] = {...}

# NEW FILTER 1: At least 4/5 profitable IS in W1..W5 (generalized to HIST_WINDOWS)
need_hist_prof = int(np.ceil(0.8 * HIST_WINDOWS))
flt_is_hist_80 = {"count": 0, "wa": 0, "wb": 0}

# NEW FILTER 2: per robustness tag, at least 80% history windows profitable in BOTH IS and IS+TAG
flt_is_rb_hist_80 = {}  # tag -> bucket dict

# NEW FILTER 3: OOS winrate bins over history window set (generalized)
winrate_bins = {
    "OOS_WR_<50": {"count": 0, "wa": 0, "wb": 0},
    "OOS_WR_50_70": {"count": 0, "wa": 0, "wb": 0},
    "OOS_WR_>70": {"count": 0, "wa": 0, "wb": 0},
}

# -------------------------------
# Regex
# -------------------------------
# Capture any robustness suffix, e.g. +ENT, +FEE, +SLI, +ENT+IND, +RB (old)
RE_LINE = re.compile(r"^\s*W(\d{2})\s+(IS|OOS)(?:\+([A-Za-z+]+))?", re.IGNORECASE)
RE_PF = re.compile(r"PF:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_WIN = re.compile(r"\bWin:\s*([+-]?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
RE_ROI = re.compile(r"ROI:\s*\$([+-]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
RE_TRADES = re.compile(r"Trades:\s*([0-9]+)", re.IGNORECASE)

# Generic key:value parser (to export "everything available" from each line)
RE_KV = re.compile(r"(\b[A-Za-z][A-Za-z0-9_]*\b):\s*(\$)?([+-]?\d[\d,]*(?:\.\d+)?)(%)?", re.IGNORECASE)

# Stop loss parsing from filename: "..._SLX..." where X can be int or float
RE_SL = re.compile(r"_SL(\d+(?:\.\d+)?)", re.IGNORECASE)

total_strategies = 0
parsed_strategies = 0
robust_tags_seen = set()

# -------------------------------
# EXTRA: >100 trades (TOTAL across all samples) + dedupe bucket
# -------------------------------
dedupe_signatures_seen = set()
dup_removed_count = 0
lt100_skipped_count = 0

flt_gt100_dedup = {"count": 0, "wa": 0, "wb": 0, "wab": 0}

# -------------------------------
# Helpers
# -------------------------------
def map_file_window_to_active(file_window: int):
    """Map file window numbers to active 1..EXPECTED_WINDOWS space; return None if outside active range."""
    if file_window < WINDOW_FILE_START or file_window > WINDOW_FILE_END:
        return None
    return file_window - WINDOW_OFFSET

def normalize_window_label_to_active(label: str):
    """
    Convert file labels like W03..W08 into active labels W01..W06 when offset is used.
    Labels outside the active range are returned unchanged.
    """
    s = str(label).strip().upper()
    m = re.match(r"^W(\d{1,3})$", s)
    if not m:
        return s
    file_w = int(m.group(1))
    active_w = map_file_window_to_active(file_w)
    if active_w is None:
        return s
    return f"W{active_w:02d}"

def ensure_rb_tag(tag: str):
    """Initialize per-tag stats/buckets the first time we see a robustness tag."""
    if tag not in stats_by_rb:
        stats_by_rb[tag] = {
            k: {
                tier: {"count": 0, "wa": 0, "wb": 0, "wab": 0}
                for tier in TIERS_RB
            }
            for k in range(1, EXPECTED_WINDOWS + 1)
        }
    if tag not in flt_is_rb_hist_80:
        flt_is_rb_hist_80[tag] = {"count": 0, "wa": 0, "wb": 0}

def expand_robustness_tags(tag_raw: str):
    """
    Expand robustness suffixes into component tags.
    Example: "ENT+IND" -> ["ENT", "IND"].
    """
    if not tag_raw:
        return []
    out = []
    seen = set()
    for part in str(tag_raw).upper().split("+"):
        p = part.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out

def bump_bucket(bucket, oos_list):
    bucket["count"] += 1
    wa = (oos_list[LIVE_IA] >= PF_OK)
    wb = (oos_list[LIVE_IB] >= PF_OK)
    if wa:
        bucket["wa"] += 1
    if wb:
        bucket["wb"] += 1

def bump_base_k(k, oos_list):
    s = stats_base[k]["IS"]
    s["count"] += 1
    wa = (oos_list[LIVE_IA] >= PF_OK)
    wb = (oos_list[LIVE_IB] >= PF_OK)
    if wa:
        s["wa"] += 1
    if wb:
        s["wb"] += 1
    if wa and wb:
        s["wab"] += 1

def bump_rb_k_tier(tag, k, tier, oos_list):
    s = stats_by_rb[tag][k][tier]
    s["count"] += 1
    wa = (oos_list[LIVE_IA] >= PF_OK)
    wb = (oos_list[LIVE_IB] >= PF_OK)
    if wa:
        s["wa"] += 1
    if wb:
        s["wb"] += 1
    if wa and wb:
        s["wab"] += 1

def parse_money_to_float(s: str) -> float:
    return float(s.replace(",", ""))

def compute_hist_trades(base_is_trades: dict, base_oos_trades: dict, start_w: int = 1, hist_windows: int = HIST_WINDOWS):
    """
    History trade count uses IS at start_w plus OOS for start_w..end_w.
    Returns (has_all, total_trades).
    """
    end_w = min(EXPECTED_WINDOWS, start_w + hist_windows - 1)
    has_is = start_w in base_is_trades
    has_hist_oos = all(w in base_oos_trades for w in range(start_w, end_w + 1))
    if not (has_is and has_hist_oos):
        return False, 0
    total = base_is_trades[start_w] + sum(base_oos_trades[w] for w in range(start_w, end_w + 1))
    return True, total

def compute_total_trades_range(base_is_trades: dict, base_oos_trades: dict, start_w: int, end_w: int):
    """
    Total trade count for dedupe over an arbitrary window range (inclusive).
    """
    if start_w > end_w:
        return False, 0
    has_is = start_w in base_is_trades
    has_all_oos = all(w in base_oos_trades for w in range(start_w, end_w + 1))
    if not (has_is and has_all_oos):
        return False, 0
    total = base_is_trades[start_w] + sum(base_oos_trades[w] for w in range(start_w, end_w + 1))
    return True, total

def compute_total_trades_all(base_is_trades: dict, base_oos_trades: dict):
    """
    Total trade count for dedupe: W01-IS plus OOS for all windows.
    Returns (has_all, total_trades).
    """
    return compute_total_trades_range(base_is_trades, base_oos_trades, 1, EXPECTED_WINDOWS)

def get_sl_from_filename_global(fname: str):
    m = RE_SL.search(fname)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def find_trade_list_file_global(strategy_dir: str):
    p1 = os.path.join(strategy_dir, "trade_list", "trade_list.csv")
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(strategy_dir, "trade_list.csv")
    if os.path.isfile(p2):
        return p2
    tl_dir = os.path.join(strategy_dir, "trade_list")
    if os.path.isdir(tl_dir):
        candidates = []
        for fn in os.listdir(tl_dir):
            fp = os.path.join(tl_dir, fn)
            if not os.path.isfile(fp):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".csv", ".tsv", ".txt"):
                pri = 0 if fn.lower() == "trade_list.csv" else (1 if ext == ".csv" else 2)
                candidates.append((pri, fp))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    return None

def load_trade_list_global(strategy_dir: str):
    fp = find_trade_list_file_global(strategy_dir)
    if fp is None:
        return None, None
    try:
        tdf = pd.read_csv(fp, sep=None, engine="python")
    except Exception:
        return None, fp

    tdf.columns = [str(c).strip() for c in tdf.columns]
    cols_lc = {c.lower(): c for c in tdf.columns}
    required = ["window", "sample", "pnl"]
    if not all(r in cols_lc for r in required):
        return None, fp

    tdf = tdf.rename(columns={
        cols_lc["window"]: "window",
        cols_lc["sample"]: "sample",
        cols_lc["pnl"]: "pnl",
    })
    if "exit_time" in cols_lc:
        tdf = tdf.rename(columns={cols_lc["exit_time"]: "exit_time"})
    if "entry_time" in cols_lc:
        tdf = tdf.rename(columns={cols_lc["entry_time"]: "entry_time"})

    tdf["window"] = tdf["window"].astype(str).str.strip().map(normalize_window_label_to_active)
    tdf["sample"] = tdf["sample"].astype(str).str.strip().str.upper()
    tdf["pnl"] = pd.to_numeric(tdf["pnl"], errors="coerce")
    tdf = tdf.dropna(subset=["pnl"])
    return tdf, fp

def _resolve_backtest_only_windows():
    start = int(Backtest_only_window_start)
    hist_n = int(Backtest_only_history_windows)
    live_n = int(Backtest_only_live_windows)
    total = hist_n + live_n

    if hist_n <= 0 or live_n <= 0:
        raise ValueError("Backtest_only_history_windows and Backtest_only_live_windows must be >= 1.")
    if total != EXPECTED_WINDOWS:
        raise ValueError(
            f"Backtest-only windows must sum to EXPECTED_WINDOWS={EXPECTED_WINDOWS}, got {hist_n}+{live_n}={total}."
        )
    if start < 1:
        raise ValueError("Backtest_only_window_start must be >= 1.")
    if (start + total - 1) > WINDOWS_IN_FILE:
        raise ValueError(
            f"Backtest_only_window_start={start} with total={total} exceeds WINDOWS_IN_FILE={WINDOWS_IN_FILE}."
        )

    hist_labels = [f"W{w:02d}" for w in range(start, start + hist_n)]
    live_labels = [f"W{w:02d}" for w in range(start + hist_n, start + hist_n + live_n)]
    all_labels = hist_labels + live_labels
    return {
        "start": start,
        "hist_labels": hist_labels,
        "live_labels": live_labels,
        "all_labels": all_labels,
        "child_offset": start - 1,
    }

def _plot_distribution_curve(values: np.ndarray, title: str, xlabel: str):
    plt.figure(figsize=(12, 6))
    if values.size:
        bins = min(40, max(8, int(np.sqrt(values.size))))
        plt.hist(values, bins=bins, density=True, alpha=0.45, color="steelblue", edgecolor="black")
        counts, edges = np.histogram(values, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        if counts.size and np.any(np.isfinite(counts)):
            smooth_window = min(7, counts.size if counts.size % 2 == 1 else max(1, counts.size - 1))
            if smooth_window >= 3:
                kernel = np.ones(smooth_window, dtype=float) / smooth_window
                smooth_counts = np.convolve(counts, kernel, mode="same")
            else:
                smooth_counts = counts
            plt.plot(centers, smooth_counts, color="navy", linewidth=2.0, label="Distribution curve")
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=0))
        plt.axvline(mean_v, color="darkgreen", linestyle="-", linewidth=2.0, label=f"Mean={mean_v:.3f}R")
        if np.isfinite(std_v) and std_v > 0:
            plt.axvline(mean_v + std_v, color="orange", linestyle="--", linewidth=1.8, label=f"+1 std={std_v:.3f}R")
            plt.axvline(mean_v - std_v, color="orange", linestyle="--", linewidth=1.8, label=f"-1 std={std_v:.3f}R")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def _load_trade_list_fast_for_windows(strategy_dir: str, keep_windows: set):
    fp = find_trade_list_file_global(strategy_dir)
    if fp is None:
        return None

    target_cols = {"window", "sample", "pnl", "exit_time", "entry_time"}
    tdf = None
    try:
        tdf = pd.read_csv(
            fp,
            usecols=lambda c: str(c).strip().lower() in target_cols,
            engine="c",
        )
    except Exception:
        try:
            tdf = pd.read_csv(
                fp,
                sep=None,
                engine="python",
                usecols=lambda c: str(c).strip().lower() in target_cols,
            )
        except Exception:
            return None

    if tdf is None or tdf.empty:
        return None

    tdf.columns = [str(c).strip() for c in tdf.columns]
    cols_lc = {c.lower(): c for c in tdf.columns}
    if not all(k in cols_lc for k in ("window", "sample", "pnl")):
        return None

    rename_map = {
        cols_lc["window"]: "window",
        cols_lc["sample"]: "sample",
        cols_lc["pnl"]: "pnl",
    }
    if "exit_time" in cols_lc:
        rename_map[cols_lc["exit_time"]] = "exit_time"
    if "entry_time" in cols_lc:
        rename_map[cols_lc["entry_time"]] = "entry_time"
    tdf = tdf.rename(columns=rename_map)

    tdf["window"] = tdf["window"].astype(str).str.strip().str.upper()
    tdf["sample"] = tdf["sample"].astype(str).str.strip().str.upper()
    tdf["pnl"] = pd.to_numeric(tdf["pnl"], errors="coerce")
    tdf = tdf.dropna(subset=["pnl"])
    if tdf.empty:
        return None

    tdf = tdf[(tdf["sample"] == "OOS") & (tdf["window"].isin(keep_windows))]
    return None if tdf.empty else tdf

def _run_backtest_all_strategies_distribution(all_matched: dict, bt_cfg: dict):
    live_set = set(bt_cfg["live_labels"])
    strategy_info = {}
    for s, txt_path in all_matched.items():
        strategy_dir = os.path.dirname(txt_path)
        sl = get_sl_from_filename_global(os.path.basename(txt_path))
        if sl is None or sl == 0:
            continue

        tdf = _load_trade_list_fast_for_windows(strategy_dir, live_set)
        if tdf is None or tdf.empty:
            continue
        live_pnl = pd.to_numeric(tdf["pnl"], errors="coerce").dropna().values.astype(float)
        if live_pnl.size == 0:
            continue

        live_r = live_pnl / (25.0 * sl)
        strategy_info[s] = {
            "live_r": live_r,
            "live_roi_r": float(np.sum(live_r)),
        }

    if not strategy_info:
        log_line("\n[Backtest_only] No strategies had usable live-proxy OOS trades for all-strategy distribution.\n")
        return

    live_roi_values = np.array(
        [v["live_roi_r"] for v in strategy_info.values() if np.isfinite(v["live_roi_r"])],
        dtype=float
    )
    pooled_live_r = np.concatenate(
        [v["live_r"] for v in strategy_info.values() if v["live_r"].size],
        axis=0
    ) if any(v["live_r"].size for v in strategy_info.values()) else np.array([], dtype=float)

    _plot_distribution_curve(
        live_roi_values,
        title=(
            f"Backtest_only: All Strategies Live ROI Distribution "
            f"({' & '.join(bt_cfg['live_labels'])})"
        ),
        xlabel="Per-strategy ROI (R)"
    )

    mean_roi = float(np.mean(live_roi_values)) if live_roi_values.size else np.nan
    std_roi = float(np.std(live_roi_values, ddof=0)) if live_roi_values.size else np.nan
    expectancy_roi = mean_roi
    pooled_expectancy = float(np.mean(pooled_live_r)) if pooled_live_r.size else np.nan

    log_line("\n[Backtest_only] All-strategies live distribution complete.")
    log_line(f"Strategies scanned (.txt): {len(all_matched)}")
    log_line(f"Strategies with usable trades: {len(strategy_info)}")
    log_line(f"Strategies with finite live ROI: {live_roi_values.size}")
    if np.isfinite(mean_roi):
        log_line(f"Live ROI mean (R): {mean_roi:.4f}")
    if np.isfinite(expectancy_roi):
        log_line(f"Live ROI expectancy (R): {expectancy_roi:.4f}")
    if np.isfinite(std_roi):
        log_line(f"Live ROI std (R): {std_roi:.4f}")
    if np.isfinite(pooled_expectancy):
        log_line(f"Pooled live expectancy (R/trade): {pooled_expectancy:.6f}")

def _run_backtest_selected_weighted_portfolio(all_matched: dict, bt_cfg: dict):
    targets = {}
    if isinstance(backtest_list, dict):
        targets = {str(k).strip(): float(v) for k, v in backtest_list.items() if str(k).strip()}
        targets = {k: v for k, v in targets.items() if v > 0}
    if not targets:
        log_line("\n[Backtest_only] Selected portfolio plot skipped: backtest_list has no positive weights.\n")
        return

    selected = [s for s in targets.keys() if s in all_matched]
    missing = [s for s in targets.keys() if s not in all_matched]
    if missing:
        log_line("\n[Backtest_only] Missing strategy .txt files for:")
        for m in missing:
            log_line(f"  - {m}")
    if not selected:
        log_line("\n[Backtest_only] Selected portfolio plot skipped: no selected strategies found on disk.\n")
        return

    total_w = float(sum(targets[s] for s in selected))
    if total_w <= 0:
        log_line("\n[Backtest_only] Selected portfolio plot skipped: invalid total weight.\n")
        return
    norm_w = {s: (targets[s] / total_w) for s in selected}

    per_strategy_trades = {}
    keep_windows = set(bt_cfg["all_labels"])
    live_first = bt_cfg["live_labels"][0]

    for s in selected:
        txt_path = all_matched[s]
        strategy_dir = os.path.dirname(txt_path)
        sl = get_sl_from_filename_global(os.path.basename(txt_path))
        if sl is None or sl == 0:
            continue

        tdf = _load_trade_list_fast_for_windows(strategy_dir, keep_windows)
        if tdf is None or tdf.empty:
            continue

        per_window_weighted = {}
        for ww in bt_cfg["all_labels"]:
            block = tdf[tdf["window"] == ww].copy()
            if block.empty:
                per_window_weighted[ww] = pd.DataFrame(columns=["sort_key", "r_w"])
                continue

            if "exit_time" in block.columns:
                block["sort_key"] = pd.to_datetime(block["exit_time"], errors="coerce")
            elif "entry_time" in block.columns:
                block["sort_key"] = pd.to_datetime(block["entry_time"], errors="coerce")
            else:
                block["sort_key"] = np.arange(len(block))

            block["r_w"] = (pd.to_numeric(block["pnl"], errors="coerce") / (25.0 * sl)) * norm_w[s]
            block = block.dropna(subset=["r_w"])
            block = block.sort_values("sort_key", kind="mergesort")
            per_window_weighted[ww] = block[["sort_key", "r_w"]]

        per_strategy_trades[s] = per_window_weighted

    if not per_strategy_trades:
        log_line("\n[Backtest_only] Selected portfolio plot skipped: no usable selected strategies.\n")
        return

    portfolio_r_parts = []
    live_split_trade_idx = None
    running_len = 0
    for ww in bt_cfg["all_labels"]:
        if ww == live_first and live_split_trade_idx is None:
            live_split_trade_idx = running_len
        win_parts = []
        for s in per_strategy_trades.keys():
            dfw = per_strategy_trades[s].get(ww)
            if dfw is not None and not dfw.empty:
                win_parts.append(dfw)
        if not win_parts:
            continue
        merged = pd.concat(win_parts, ignore_index=True).sort_values("sort_key", kind="mergesort")
        rvals = pd.to_numeric(merged["r_w"], errors="coerce").dropna().values.astype(float)
        if rvals.size:
            portfolio_r_parts.append(rvals)
            running_len += int(rvals.size)

    if not portfolio_r_parts:
        log_line("\n[Backtest_only] Selected portfolio has zero OOS trades after filtering.\n")
        return

    portfolio_r = np.concatenate(portfolio_r_parts)
    equity_r = np.cumsum(portfolio_r)
    x = np.arange(1, equity_r.size + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(x, equity_r, color="navy", linewidth=1.8, label="Portfolio Equity (R)")
    if live_split_trade_idx is not None and live_split_trade_idx < equity_r.size:
        plt.axvline(
            x=live_split_trade_idx + 1,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Live proxies start ({live_first})"
        )
    plt.title("Backtest_only: Selected Weighted Portfolio Equity Curve")
    plt.xlabel("Trade index (continuous across selected windows)")
    plt.ylabel("Cumulative R")
    plt.legend()
    plt.tight_layout()
    plt.show()

    log_line("\n[Backtest_only] Selected weighted portfolio plot complete.")
    log_line(f"Selected strategies used: {len(per_strategy_trades)}")
    log_line(f"Total OOS trades: {equity_r.size}")
    log_line(f"Final equity (R): {equity_r[-1]:.2f}")

def _run_backtest_all_portfolios_distribution(bt_cfg: dict):
    script_path = os.path.abspath(__file__) if "__file__" in globals() else None
    if not script_path or not os.path.isfile(script_path):
        log_line("[Backtest_only] Could not resolve script path for portfolio distribution child run.")
        return

    with tempfile.NamedTemporaryFile(prefix="wfo_port_roi_", suffix=".json", delete=False) as tf:
        out_json = tf.name

    env = os.environ.copy()
    env[ENV_FORCE_FULL_PIPELINE] = "1"
    env[ENV_BACKTEST_PORT_ROI_OUT] = out_json
    env[ENV_BACKTEST_SKIP_XLSX] = "1"
    env[ENV_WINDOW_OFFSET_OVERRIDE] = str(bt_cfg["child_offset"])
    env.setdefault("PYTHONIOENCODING", "utf-8")

    log_line(
        f"\n[Backtest_only] Running full pipeline child for all-portfolios distribution "
        f"(test {int(SELECT_TEST_NUMBER)}, offset={bt_cfg['child_offset']})..."
    )
    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(script_path) or None,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
    )

    if proc.returncode != 0:
        log_line("\n[Backtest_only] Child run failed while generating portfolio distribution.")
        if proc.stderr and proc.stderr.strip():
            log_line(proc.stderr[-4000:])
        return

    if not os.path.isfile(out_json):
        log_line("\n[Backtest_only] Child run completed but did not output portfolio ROI data.")
        return

    try:
        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log_line(f"\n[Backtest_only] Failed to read portfolio ROI data: {e}")
        return
    finally:
        try:
            os.remove(out_json)
        except Exception:
            pass

    values = np.array(payload.get("live_roi_r", []), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        log_line("\n[Backtest_only] No finite portfolio live ROI values were produced.")
        return

    _plot_distribution_curve(
        values,
        title=(
            f"Backtest_only: All Portfolios Live ROI Distribution "
            f"(selected test {int(SELECT_TEST_NUMBER)})"
        ),
        xlabel="Per-portfolio live ROI (R)"
    )
    log_line("\n[Backtest_only] All-portfolios distribution complete.")
    log_line(f"Portfolios included: {values.size}")
    log_line(f"Mean ROI (R): {float(np.mean(values)):.4f}")
    log_line(f"Expectancy ROI (R): {float(np.mean(values)):.4f}")
    log_line(f"Std ROI (R): {float(np.std(values, ddof=0)):.4f}")

def run_backtest_only():
    bt_cfg = _resolve_backtest_only_windows()
    log_line(
        f"\n[Backtest_only] Window split configuration:\n"
        f"  - History block: {bt_cfg['hist_labels'][0]}..{bt_cfg['hist_labels'][-1]}\n"
        f"  - Live-proxy block: {bt_cfg['live_labels'][0]}..{bt_cfg['live_labels'][-1]}\n"
        f"  - Child offset used for full-pipeline child runs: {bt_cfg['child_offset']}"
    )

    all_matched = {}
    all_duplicates = set()
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
            stem = os.path.splitext(file)[0]
            path = os.path.join(subdir, file)
            if stem in all_matched:
                all_duplicates.add(stem)
            else:
                all_matched[stem] = path

    if not all_matched:
        log_line("\n[Backtest_only] No strategy .txt files were found.\n")
        return

    if all_duplicates:
        log_line("\n[Backtest_only] Duplicate strategy names found (using first match):")
        for d in sorted(all_duplicates):
            log_line(f"  - {d}")

    ran_any = False
    if Backtest_all_strategies_distribution_curve:
        _run_backtest_all_strategies_distribution(all_matched, bt_cfg)
        ran_any = True
    if Backtest_only_plot_selected_portfolio:
        _run_backtest_selected_weighted_portfolio(all_matched, bt_cfg)
        ran_any = True
    if Backtest_all_portfolios_distribution_curve:
        _run_backtest_all_portfolios_distribution(bt_cfg)
        ran_any = True

    if not ran_any:
        log_line(
            "\n[Backtest_only] Nothing to run. Enable at least one option:\n"
            "  - Backtest_only_plot_selected_portfolio\n"
            "  - Backtest_all_strategies_distribution_curve\n"
            "  - Backtest_all_portfolios_distribution_curve\n"
        )

if Backtest_only:
    run_backtest_only()
    raise SystemExit(0)

# -------------------------------
# Walk files (PASS 1: stats + heatmaps + pipelines)
# -------------------------------
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if not file.endswith(".txt"):
            continue

        total_strategies += 1
        path = os.path.join(subdir, file)

        # Base (no robustness suffix)
        is_pf = {}
        oos_pf = {}
        oos_win = {}
        is_roi = {}
        oos_roi = {}
        is_trades = {}
        oos_trades = {}

        # Robustness variants: tag -> {window -> pf}
        rb_is_pf = defaultdict(dict)
        rb_oos_pf = defaultdict(dict)

        # for dedupe signature: include all matched lines (base + robustness)
        matched_lines_norm = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = RE_LINE.match(line)
                if not m:
                    continue

                w_file = int(m.group(1))
                w = map_file_window_to_active(w_file)
                if w is None:
                    continue
                # normalize for signature (ignore lines outside active window range)
                matched_lines_norm.append(" ".join(line.strip().split()))

                phase = m.group(2).upper()
                tag_raw = m.group(3)
                tag = tag_raw.upper() if tag_raw else None  # raw label (for diagnostics/exports)
                tag_variants = expand_robustness_tags(tag_raw)

                pf_m = RE_PF.search(line)
                if not pf_m:
                    continue
                pf = float(pf_m.group(1))

                roi_m = RE_ROI.search(line)
                win_m = RE_WIN.search(line)

                tr_m = RE_TRADES.search(line)
                trades = int(tr_m.group(1)) if tr_m else None

                if tag is None:
                    # base stream
                    if phase == "IS":
                        is_pf[w] = pf
                        if trades is not None:
                            is_trades[w] = trades  # track IS trades for all windows (needed for live-mode recheck)
                        if roi_m:
                            is_roi[w] = parse_money_to_float(roi_m.group(1))

                    elif phase == "OOS":
                        oos_pf[w] = pf
                        if trades is not None:
                            oos_trades[w] = trades
                        if roi_m:
                            oos_roi[w] = parse_money_to_float(roi_m.group(1))
                        if win_m:
                            oos_win[w] = float(win_m.group(1))

                else:
                    # robustness stream
                    for t in tag_variants:
                        robust_tags_seen.add(t)
                        ensure_rb_tag(t)
                        if phase == "IS":
                            rb_is_pf[t][w] = pf
                        elif phase == "OOS":
                            rb_oos_pf[t][w] = pf

        # Early skip: require W01 IS and OOS history trades; all other history windows use OOS only
        has_hist_trades, total_trades_hist = compute_hist_trades(is_trades, oos_trades)
        if not has_hist_trades:
            continue
        if total_trades_hist < 100:
            lt100_skipped_count += 1
            continue

        # Require all windows present for base streams used by logic
        try:
            is_list = [is_pf[i] for i in range(1, EXPECTED_WINDOWS + 1)]
            oos_list = [oos_pf[i] for i in range(1, EXPECTED_WINDOWS + 1)]
        except KeyError:
            continue

        parsed_strategies += 1

        # ---------- >100 trades TOTAL across all base samples (IS & OOS across all windows) + remove duplicates ----------
        has_all_trades, total_trades_all_samples = compute_total_trades_all(is_trades, oos_trades)
        if has_all_trades and total_trades_all_samples >= 100:
            sig = "\n".join(sorted(matched_lines_norm))  # "exact performance across samples"
            if sig in dedupe_signatures_seen:
                dup_removed_count += 1
            else:
                dedupe_signatures_seen.add(sig)

                flt_gt100_dedup["count"] += 1
                wa = (oos_list[LIVE_IA] >= PF_OK)
                wb = (oos_list[LIVE_IB] >= PF_OK)
                if wa:
                    flt_gt100_dedup["wa"] += 1
                if wb:
                    flt_gt100_dedup["wb"] += 1
                if wa and wb:
                    flt_gt100_dedup["wab"] += 1

        # ---------- BASE OUTPUT LOGIC (kept) ----------
        is_prof_idx_all = [i for i, pfv in enumerate(is_list) if pfv >= PF_OK]  # 0..EXPECTED_WINDOWS-1
        k = len(is_prof_idx_all)
        if k < 1 or k > EXPECTED_WINDOWS:
            continue

        bump_base_k(k, oos_list)

        # ---------- PER-ROBUSTNESS OUTPUT LOGIC (kept) ----------
        is_prof_w_nums = [i + 1 for i in is_prof_idx_all]
        tags_in_file = set(rb_is_pf.keys()) & set(rb_oos_pf.keys())

        for tag in tags_in_file:
            cond_is_tag = all(rb_is_pf[tag].get(w, -1e18) >= PF_OK for w in is_prof_w_nums)
            if not cond_is_tag:
                continue

            bump_rb_k_tier(tag, k, "IS_ISRB", oos_list)

            cond_oos_base = all(oos_list[w - 1] >= PF_OK for w in is_prof_w_nums)
            if not cond_oos_base:
                continue

            bump_rb_k_tier(tag, k, "IS_ISRB_OOS", oos_list)

            cond_oos_tag = all(rb_oos_pf[tag].get(w, -1e18) >= PF_OK for w in is_prof_w_nums)
            if not cond_oos_tag:
                continue

            bump_rb_k_tier(tag, k, "IS_ISRB_OOS_OOSRB", oos_list)

        # ---------- NEW FILTERS (generalized to history windows) ----------
        is_hist_prof = sum(1 for i in HIST_IDXS if is_list[i] >= PF_OK)
        if is_hist_prof >= need_hist_prof:
            bump_bucket(flt_is_hist_80, oos_list)

        # Per-tag: history windows profitable IS and IS+TAG
        for tag in tags_in_file:
            rb_hist_prof = 0
            rb_is_map = rb_is_pf[tag]
            for wi in range(1, HIST_WINDOWS + 1):
                if is_pf.get(wi, -1e18) >= PF_OK and rb_is_map.get(wi, -1e18) >= PF_OK:
                    rb_hist_prof += 1
            if rb_hist_prof >= need_hist_prof:
                bump_bucket(flt_is_rb_hist_80[tag], oos_list)

        # Winrate bins based on AVG Win% across OOS history windows
        if all(w in oos_win for w in HIST_W_NUMS):
            avg_wr = sum(oos_win[w] for w in HIST_W_NUMS) / float(HIST_WINDOWS)
            if avg_wr < 50.0:
                bump_bucket(winrate_bins["OOS_WR_<50"], oos_list)
            elif 50.0 <= avg_wr <= 70.0:
                bump_bucket(winrate_bins["OOS_WR_50_70"], oos_list)
            else:
                bump_bucket(winrate_bins["OOS_WR_>70"], oos_list)

# -------------------------------
# RUN SUMMARY (base parsing and window geometry)
# -------------------------------
log_line("\n==================== PARSING SUMMARY ====================\n")
log_line(f"Total strategy text files scanned: {total_strategies}")
log_line(f"Strategies with complete required base windows: {parsed_strategies}")
if LIVE_MODE:
    log_line(
        "Active window interpretation: "
        f"EXPECTED_WINDOWS={EXPECTED_WINDOWS}, WINDOWS_IN_FILE={WINDOWS_IN_FILE}, "
        f"ACTIVE_FILE_WINDOWS=W{WINDOW_FILE_START:02d}..W{WINDOW_FILE_END:02d}, "
        f"LIVE_MODE_START=W{LIVE_MODE_START:02d}, LIVE_PROXIES=NONE (LIVE_MODE)\n"
    )
else:
    log_line(
        "Active window interpretation: "
        f"EXPECTED_WINDOWS={EXPECTED_WINDOWS}, WINDOWS_IN_FILE={WINDOWS_IN_FILE}, "
        f"ACTIVE_FILE_WINDOWS=W{WINDOW_FILE_START:02d}..W{WINDOW_FILE_END:02d}, "
        f"HISTORY=W1..W{HIST_WINDOWS}, LIVE_PROXIES=W{LIVE_WA}&W{LIVE_WB}\n"
    )
log_line(f"Window-count detection source/file: {WINDOWS_DETECTED_FROM}")

# This block answers: "if exactly k IS windows are profitable, how often are live proxies profitable?"
for k in range(1, EXPECTED_WINDOWS + 1):
    s = stats_base[k]["IS"]
    log_line(f"\n===== BASE FILTER: EXACTLY {k} PROFITABLE IS WINDOWS =====")
    if s["count"] == 0:
        log_line("No strategies matched this exact-k condition.\n")
        continue

    wa = 100.0 * s["wa"] / s["count"]
    wb = 100.0 * s["wb"] / s["count"]
    wab = 100.0 * s["wab"] / s["count"]

    log_line(
        f"Matched strategies: {s['count']}\n"
        f"  - Live proxy W{LIVE_WA} profitable rate: {wa:.2f}%\n"
        f"  - Live proxy W{LIVE_WB} profitable rate: {wb:.2f}%\n"
        f"  - Both live proxies profitable rate: {wab:.2f}%\n"
    )

# -------------------------------
# PRINT RESULTS (new filters)
# -------------------------------
def print_bucket(title, bucket):
    log_line(f"\n{title}")
    if bucket["count"] == 0:
        log_line("  - Matched strategies: 0")
        log_line(f"  - Live proxy W{LIVE_WA} profitable rate: 0.00%")
        log_line(f"  - Live proxy W{LIVE_WB} profitable rate: 0.00%\n")
        return
    wa = 100.0 * bucket["wa"] / bucket["count"]
    wb = 100.0 * bucket["wb"] / bucket["count"]
    log_line(f"  - Matched strategies: {bucket['count']}")
    log_line(f"  - Live proxy W{LIVE_WA} profitable rate: {wa:.2f}%")
    log_line(f"  - Live proxy W{LIVE_WB} profitable rate: {wb:.2f}%\n")

log_line("\n=== ADDITIONAL FILTERS AND WHAT THEY MEAN ===\n")
print_bucket(f"At least {need_hist_prof}/{HIST_WINDOWS} history windows profitable IS (W1..W{HIST_WINDOWS}):", flt_is_hist_80)

if flt_is_rb_hist_80:
    log_line("=== PER-ROBUSTNESS HISTORY FILTER (IS & IS+TAG) ===\n")
    for tag in sorted(flt_is_rb_hist_80.keys()):
        print_bucket(
            f"At least {need_hist_prof}/{HIST_WINDOWS} history windows profitable IS and IS+{tag} (W1..W{HIST_WINDOWS}):",
            flt_is_rb_hist_80[tag]
        )

log_line(f"=== OOS WINRATE BINS (average OOS win% over history W1..W{HIST_WINDOWS}) ===\n")
print_bucket("Winrate in OOS history < 50%:", winrate_bins["OOS_WR_<50"])
print_bucket("Winrate in OOS history 50%-70%:", winrate_bins["OOS_WR_50_70"])
print_bucket("Winrate in OOS history > 70%:", winrate_bins["OOS_WR_>70"])

# -------------------------------
# FINAL PRINTS (requested format) + BETA LOWER BOUND TABLE
# -------------------------------
def _pct(n, d):
    return 0.0 if d == 0 else (100.0 * n / d)

def beta_lower_bound_95(rate_pct: float, n: int) -> float:
    """
    Beta-Binomial (uniform prior):
      posterior = Beta(1 + k, 1 + n - k)
      k = round(n * rate/100)
      95% lower bound = beta.ppf(0.05, a, b) * 100
    """
    if n <= 0:
        return 0.0
    k = int(round(n * (rate_pct / 100.0)))
    k = max(0, min(k, n))
    a = 1 + k
    b = 1 + (n - k)
    lb = float(beta_dist.ppf(0.05, a, b)) * 100.0
    return 0.0 if not np.isfinite(lb) else lb


log_line("\n------------------- BASE QUALITY GATE: >100 TRADES + DEDUP -------------------")
n0 = flt_gt100_dedup["count"]
wa0 = _pct(flt_gt100_dedup["wa"], n0)
wb0 = _pct(flt_gt100_dedup["wb"], n0)
wab0 = _pct(flt_gt100_dedup["wab"], n0)

log_line(f"  - Matched strategies: {n0}")
log_line(f"  - Live proxy W{LIVE_WA} profitable rate: {wa0:.2f}%")
log_line(f"  - Live proxy W{LIVE_WB} profitable rate: {wb0:.2f}%")
log_line(f"  - Both live proxies profitable rate: {wab0:.2f}%")
log_line("  - Dedup rule: exact same performance signature across samples.")

_add_pipeline_row("Strategies with >100 trades total (base IS+OOS across all windows) + deduped", f"W{LIVE_WA}", n0, wa0)
_add_pipeline_row("Strategies with >100 trades total (base IS+OOS across all windows) + deduped", f"W{LIVE_WB}", n0, wb0)
_add_pipeline_row("Strategies with >100 trades total (base IS+OOS across all windows) + deduped", f"W{LIVE_WA}&W{LIVE_WB}", n0, wab0)

if robust_tags_seen:
    for tag in sorted(robust_tags_seen):
        any_count = any(stats_by_rb[tag][k][t]["count"] > 0 for k in range(1, EXPECTED_WINDOWS + 1) for t in TIERS_RB)
        if not any_count:
            continue

        b_2_oos = _agg_rb_bucket(tag, "IS_ISRB_OOS", k_min=2)
        _print_block_and_collect(f"\nStrategies with 2 or more profitable windows (base) + IS+{tag} + OOS", b_2_oos)

        b_2_oos_rb = _agg_rb_bucket(tag, "IS_ISRB_OOS_OOSRB", k_min=2)
        _print_block_and_collect(f"Strategies with 2 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}", b_2_oos_rb)

        b_3_oos = _agg_rb_bucket(tag, "IS_ISRB_OOS", k_min=3)
        _print_block_and_collect(f"\nStrategies with 3 or more profitable windows (base) + IS+{tag} + OOS", b_3_oos)

        b_3_oos_rb = _agg_rb_bucket(tag, "IS_ISRB_OOS_OOS_OOSRB" if False else "IS_ISRB_OOS_OOSRB", k_min=3)
        _print_block_and_collect(f"Strategies with 3 or more profitable windows (base) + IS+{tag} + OOS + OOS+{tag}", b_3_oos_rb)

        if EXPECTED_WINDOWS >= 4:
            b_4_oos = _agg_rb_bucket(tag, "IS_ISRB_OOS", k_exact=4)
            _print_block_and_collect(f"\nStrategies with 4 profitable windows (base) + IS+{tag} + OOS", b_4_oos)

            b_4_oos_rb = _agg_rb_bucket(tag, "IS_ISRB_OOS_OOSRB", k_exact=4)
            _print_block_and_collect(f"Strategies with 4 profitable windows (base) + IS+{tag} + OOS + OOS+{tag}", b_4_oos_rb)


# =============================================================================
# FULL TEST ID MAPPING (printed again here with complete data from file walk)
# The early pre-print may have been partial if robustness tags were not yet
# found. This version always reflects all tests discovered from actual files.
# =============================================================================
log_line("\n==================== TEST ID MAPPING (complete) ====================\n")
log_line("Each test maps to one pipeline/filter definition used for export and portfolio construction.")
log_line("Test selection rule: only robustness-funnel tests (IS+TAG / OOS+TAG pipelines) are eligible for export.\n")
if not test_id_reverse:
    log_line("  (No tests were built — no strategy .txt files were found or parsed.)\n")
else:
    for _tid in sorted(test_id_reverse.keys(), key=lambda x: int(x.split()[1])):
        _pname = test_id_reverse[_tid]
        _pdesc = parse_pipeline_descriptor(_pname)
        _eligible = "YES" if _pdesc["type"] == "RB_FUNNEL" else "NO"
        log_line(f"{_tid}: {_pname.strip()}")
        log_line(f"    Meaning: {_describe_pipeline_human_readable(_pname)}")
        log_line(f"    Requirements: {_pipeline_requirements_human_readable(_pname)}")
        log_line(f"    Eligible for manual export selection (robustness-required): {_eligible}")
        if _pdesc["type"] == "RB_FUNNEL" and _pdesc["tag"] not in robust_tags_seen:
            log_line(f"    *** WARNING: tag '{_pdesc['tag']}' not found in any strategy file — this test will have 0 matches.")

# =============================================================================
# MANUAL TEST SELECTION
# - SELECT_TEST_NUMBER chooses exactly one pipeline from the mapping above.
# - BEST_METRIC is the metric reported for that selected test.
# - The export uses only strategies that pass the selected pipeline.
# - If the chosen test is not robustness-eligible, export is skipped on purpose.
# =============================================================================
SELECT_TEST_ID = f"test {int(SELECT_TEST_NUMBER)}"
BEST_METRIC = f"W{LIVE_WA}&W{LIVE_WB}"

_TAG_TO_SWITCH = {
    "ENT": "ENTRY_DRIFT",
    "IND": "INDICATOR_VARIANCE",
    "FEE": "FEE_SHOCK",
    "SLI": "SLIPPAGE_SHOCK",
}

if SELECT_TEST_ID not in test_id_reverse:
    log_line(f"\n[EXPORT SKIPPED] {SELECT_TEST_ID} does not exist. Check the TEST ID MAPPING above and set SELECT_TEST_NUMBER accordingly.\n")
else:
    selected_pipeline_name = test_id_reverse[SELECT_TEST_ID].strip()
    selected_desc = parse_pipeline_descriptor(selected_pipeline_name)
    if selected_desc["type"] != "RB_FUNNEL":
        log_line(
            f"\n[EXPORT SKIPPED] {SELECT_TEST_ID} is not a robustness-funnel test.\n"
            "Choose a test whose mapping says "
            "'Eligible for manual export selection (robustness-required): YES'.\n"
        )
        raise SystemExit(0)

    # Graceful check: verify the required robustness tag actually exists in the data
    _required_tag = selected_desc["tag"]
    if _required_tag not in robust_tags_seen:
        _switch = _TAG_TO_SWITCH.get(_required_tag, f"the switch that produces +{_required_tag} lines")
        log_line(f"\n[EXPORT SKIPPED] {SELECT_TEST_ID} requires '{_required_tag}' robustness data,")
        log_line(f"but no strategy files contain IS+{_required_tag} / OOS+{_required_tag} lines.")
        log_line(f"To fix this, enable the following switch in run_strategies.py and re-run the backtester:")
        log_line(f"    {_switch} = True")
        log_line(f"Robustness tags found in current data: {sorted(robust_tags_seen) or '(none)'}\n")
        raise SystemExit(0)

    # Find the one row we care about: (selected test, BEST_METRIC)
    selected_rows = [r for r in pipeline_rows if (r["pipeline"] == SELECT_TEST_ID and r["metric"] == BEST_METRIC)]
    if not selected_rows:
        log_line(f"\n[EXPORT SKIPPED] No pipeline row found for {SELECT_TEST_ID} metric {BEST_METRIC}.\n")
    else:
        selected_row = selected_rows[0]
        log_line("\n==================== SELECTED TEST (MANUAL) ====================\n")
        log_line(f"Selected test id: {SELECT_TEST_ID}")
        log_line(f"Pipeline definition: {selected_pipeline_name}")
        log_line(f"Pipeline meaning: {_describe_pipeline_human_readable(selected_pipeline_name)}")
        log_line(f"Pipeline requirements: {_pipeline_requirements_human_readable(selected_pipeline_name)}")
        log_line(f"Reported metric: {BEST_METRIC} (both live proxies profitable)")
        log_line(f"Lower bound (%): {selected_row['lower_bound']:.2f}")
        log_line(f"Point estimate (%): {selected_row['point_est']:.2f}")
        log_line(f"Sample size n: {selected_row['n']}")

        desc = selected_desc
        if desc["type"] == "UNKNOWN":
            log_line(f"\n[EXPORT SKIPPED] Could not parse selected pipeline name into a filter.\nPipeline: {selected_pipeline_name}\n")
        else:
            # ---- Helpers for export parsing
            def parse_metrics_from_line(line: str) -> dict:
                out = {}
                for mm in RE_KV.finditer(line):
                    key = mm.group(1)
                    has_dollar = (mm.group(2) is not None)
                    num_s = mm.group(3)
                    has_pct = (mm.group(4) is not None)
                    try:
                        val = float(num_s.replace(",", ""))
                    except ValueError:
                        continue

                    out[key] = val
                    out[f"__is_money__{key}"] = bool(has_dollar)
                    out[f"__is_pct__{key}"] = bool(has_pct)
                return out

            def get_sl_from_filename(fname: str):
                m = RE_SL.search(fname)
                if not m:
                    return None
                try:
                    return float(m.group(1))
                except ValueError:
                    return None

            def normalize_money(val: float, sl: float):
                if sl is None or sl == 0:
                    return None
                return val / (25.0 * sl)

            def find_trade_list_file(strategy_dir: str):
                """
                Finds the trade list file for a strategy.
                Supports:
                - strategy_dir/trade_list/trade_list.csv (your stated standard)
                - strategy_dir/trade_list.csv (some exports do this)
                - fallback: any .csv/.txt/.tsv inside strategy_dir/trade_list
                """
                # 1) Most common: ./trade_list/trade_list.csv
                p1 = os.path.join(strategy_dir, "trade_list", "trade_list.csv")
                if os.path.isfile(p1):
                    return p1

                # 2) Alternate: ./trade_list.csv
                p2 = os.path.join(strategy_dir, "trade_list.csv")
                if os.path.isfile(p2):
                    return p2

                # 3) Fallback: look inside ./trade_list for any delimited file
                tl_dir = os.path.join(strategy_dir, "trade_list")
                if os.path.isdir(tl_dir):
                    candidates = []
                    for fn in os.listdir(tl_dir):
                        fp = os.path.join(tl_dir, fn)
                        if not os.path.isfile(fp):
                            continue
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in (".csv", ".tsv", ".txt"):
                            # prefer exact name first, then csv, then others
                            pri = 0 if fn.lower() == "trade_list.csv" else (1 if ext == ".csv" else 2)
                            candidates.append((pri, fp))
                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        return candidates[0][1]

                return None

            def edge_lower_bound(r_vals: np.ndarray, n_boot: int = 500, alpha: float = 0.05) -> float:
                """Bootstrap lower bound of mean/std at given alpha (default 5%)."""
                if r_vals is None or r_vals.size < 2:
                    return np.nan
                r_clean = r_vals[np.isfinite(r_vals)]
                if r_clean.size < 2 or np.nanstd(r_clean, ddof=1) == 0:
                    return np.nan
                rng = np.random.default_rng()
                edges = []
                for _ in range(n_boot):
                    sample = rng.choice(r_clean, size=r_clean.size, replace=True)
                    std = np.std(sample, ddof=1)
                    if std == 0:
                        continue
                    edges.append(np.mean(sample) / std)
                if not edges:
                    return np.nan
                return float(np.nanpercentile(edges, alpha * 100.0))

            def compute_trade_metrics(df_trades: pd.DataFrame, sl: float):
                """
                df_trades must have column 'pnl' in dollars.
                Returns: ROI_R, PF, EdgeLB, MaxDD_R, n_trades, equity_curve_R (np.array)
                """
                if df_trades is None or df_trades.empty:
                    return {
                        "ROI_R": np.nan,
                        "PF": np.nan,
                        "MaxDD_R": np.nan,
                        "Trades": 0,
                        "Equity_R": np.array([], dtype=float)
                    }

                pnl = pd.to_numeric(df_trades["pnl"], errors="coerce").dropna().values.astype(float)
                if pnl.size == 0:
                    return {
                        "ROI_R": np.nan,
                        "PF": np.nan,
                        "MaxDD_R": np.nan,
                        "Trades": 0,
                        "Equity_R": np.array([], dtype=float)
                    }

                denom = (25.0 * sl) if (sl is not None and sl != 0) else None
                r = pnl / denom if denom else pnl * np.nan

                equity = np.cumsum(r)
                peak = np.maximum.accumulate(equity) if equity.size else equity
                dd = (peak - equity) if equity.size else equity
                maxdd = float(np.max(dd)) if dd.size else np.nan

                gross_profit = float(np.sum(pnl[pnl > 0])) if np.any(pnl > 0) else 0.0
                gross_loss = float(np.sum(pnl[pnl < 0])) if np.any(pnl < 0) else 0.0
                pf = np.inf if gross_loss == 0.0 and gross_profit > 0 else (0.0 if gross_loss == 0.0 else (gross_profit / abs(gross_loss)))

                return {
                    "ROI_R": float(np.nansum(r)),
                    "PF": float(pf) if np.isfinite(pf) else float("inf"),
                    "MaxDD_R": float(maxdd),
                    "Trades": int(pnl.size),
                    "Equity_R": equity
                }

            def strategy_passes_pipeline(desc: dict, base_is_pf: dict, base_oos_pf: dict, rb_is_pf: dict, rb_oos_pf: dict,
                                         total_trades_all_samples: int, sig: str, seen_sig: set, window_offset: int = 0) -> bool:
                def is_unique(sig_):
                    if sig_ in seen_sig:
                        return False
                    seen_sig.add(sig_)
                    return True

                if desc["type"] == "GT100_DEDUP":
                    if total_trades_all_samples < 100:
                        return False
                    return is_unique(sig)

                if desc["type"] == "RB_FUNNEL":
                    tag = desc["tag"]
                    start_w = 1 + max(0, int(window_offset))
                    if start_w > EXPECTED_WINDOWS:
                        return False
                    is_prof_w = [w for w in range(start_w, EXPECTED_WINDOWS + 1) if base_is_pf.get(w, -1e18) >= PF_OK]
                    k = len(is_prof_w)

                    if desc["k_mode"] == "MIN":
                        if k < desc["k"]:
                            return False
                    else:
                        if k != desc["k"]:
                            return False

                    if tag not in rb_is_pf:
                        return False
                    if desc["with_oos_rb"] and (tag not in rb_oos_pf):
                        return False

                    if not all(rb_is_pf[tag].get(w, -1e18) >= PF_OK for w in is_prof_w):
                        return False
                    if not all(base_oos_pf.get(w, -1e18) >= PF_OK for w in is_prof_w):
                        return False
                    if desc["with_oos_rb"]:
                        if not all(rb_oos_pf[tag].get(w, -1e18) >= PF_OK for w in is_prof_w):
                            return False
                    return True

                return False

            # ---- PASS 2: collect strategies that pass SELECTED pipeline + export everything we can read
            passing_rows = []
            passing_meta_all = []  # for trade_list metric sheets and equity plotting
            dedupe_seen_pass2 = set()  # only used if selected pipeline is GT100_DEDUP
            dedupe_seen_live = set()   # separate dedupe set for live-mode reapplication

            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if not file.endswith(".txt"):
                        continue

                    path = os.path.join(subdir, file)
                    sl = get_sl_from_filename(file)

                    # Data stores
                    base_is_pf = {}
                    base_oos_pf = {}
                    base_is_trades = {}
                    base_oos_trades = {}

                    rb_is_pf = defaultdict(dict)
                    rb_oos_pf = defaultdict(dict)

                    matched_lines_norm = []
                    metrics_by_bucket = {}  # (w, phase, tag) -> metrics dict

                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            m = RE_LINE.match(line)
                            if not m:
                                continue

                            w_file = int(m.group(1))
                            w = map_file_window_to_active(w_file)
                            if w is None:
                                continue
                            matched_lines_norm.append(" ".join(line.strip().split()))
                            phase = m.group(2).upper()
                            tag_raw = m.group(3)
                            tag = tag_raw.upper() if tag_raw else None  # raw label used in exported column names
                            tag_variants = expand_robustness_tags(tag_raw)

                            met = parse_metrics_from_line(line)
                            metrics_by_bucket[(w, phase, tag)] = met

                            pf_m = RE_PF.search(line)
                            if pf_m:
                                pf = float(pf_m.group(1))
                                if tag is None:
                                    if phase == "IS":
                                        base_is_pf[w] = pf
                                    else:
                                        base_oos_pf[w] = pf
                                else:
                                    for t in tag_variants:
                                        if phase == "IS":
                                            rb_is_pf[t][w] = pf
                                        else:
                                            rb_oos_pf[t][w] = pf

                            tr_m = RE_TRADES.search(line)
                            if tr_m and tag is None:
                                tr = int(tr_m.group(1))
                                if phase == "IS":
                                    base_is_trades[w] = tr
                                elif phase == "OOS":
                                    base_oos_trades[w] = tr

                    # Evaluate both the original pipeline and the live-mode reapplication (skip first 2 windows)
                    std_hist_ok, total_hist_trades = compute_hist_trades(base_is_trades, base_oos_trades)
                    live_hist_ok, total_live_hist_trades = compute_hist_trades(base_is_trades, base_oos_trades, start_w=LIVE_MODE_START)

                    std_windows_ok = all(w in base_is_pf for w in range(1, EXPECTED_WINDOWS + 1)) and all(w in base_oos_pf for w in range(1, EXPECTED_WINDOWS + 1))
                    live_windows_ok = bool(LIVE_MODE_WINDOWS) and all(w in base_is_pf for w in LIVE_MODE_WINDOWS) and all(w in base_oos_pf for w in LIVE_MODE_WINDOWS)

                    has_all_trades_std, total_trades_all_samples = compute_total_trades_all(base_is_trades, base_oos_trades) if std_windows_ok else (False, 0)
                    has_all_trades_live, total_trades_live_range = compute_total_trades_range(base_is_trades, base_oos_trades, LIVE_MODE_START, EXPECTED_WINDOWS) if live_windows_ok else (False, 0)

                    sig = "\n".join(sorted(matched_lines_norm))

                    passes_standard = False
                    passes_live_mode = False

                    if std_hist_ok and total_hist_trades >= 100 and std_windows_ok:
                        passes_standard = strategy_passes_pipeline(
                            desc, base_is_pf, base_oos_pf, rb_is_pf, rb_oos_pf,
                            total_trades_all_samples if has_all_trades_std else 0,
                            sig, dedupe_seen_pass2
                        )

                    if live_windows_ok and live_hist_ok and total_live_hist_trades >= 100:
                        passes_live_mode = strategy_passes_pipeline(
                            desc, base_is_pf, base_oos_pf, rb_is_pf, rb_oos_pf,
                            total_trades_live_range if has_all_trades_live else 0,
                            sig, dedupe_seen_live,
                            window_offset=LIVE_MODE_SKIP
                        )

                    passes_selected = passes_live_mode if LIVE_MODE else passes_standard

                    if not passes_selected:
                        continue

                    if passes_selected:
                        row = {
                            "StrategyFile": file,
                            "FullPath": path,
                            "StopLoss_SL": sl,
                            "TotalTrades_AllBaseSamples": total_trades_all_samples,
                        }

                        for (w, phase, tag), met in metrics_by_bucket.items():
                            base_label = f"W{w:02d}_{phase}" + (f"+{tag}" if tag else "")
                            for kkey, v in met.items():
                                if kkey.startswith("__is_money__") or kkey.startswith("__is_pct__"):
                                    continue

                                col = f"{base_label}_{kkey}"
                                row[col] = v

                                is_money = bool(met.get(f"__is_money__{kkey}", False))
                                if is_money:
                                    rn = normalize_money(v, sl)
                                    row[f"{base_label}_{kkey}_Rnorm"] = rn

                        passing_rows.append(row)

                    passing_meta_all.append({
                        "strategy_dir": os.path.dirname(path),
                        "strategy_file": file,
                        "txt_path": path,
                        "sl": sl,
                        "passed_standard": passes_selected,
                        "passed_live_mode": passes_live_mode,
                    })

            if not passing_rows and not any(m.get("passed_standard") for m in passing_meta_all):
                log_line("\n[EXPORT] No strategies passed the selected test.\n")
            else:
                # =========================
                # NEW: Trade-list derived metric sheets + equity curves
                # =========================
                def load_trade_list(strategy_dir: str):
                    """
                    Loads trade_list with delimiter auto-detection.
                    Many of your files are named .csv but are actually tab-separated, so we use sep=None.
                    Returns (df, filepath)
                    """
                    fp = find_trade_list_file(strategy_dir)
                    if fp is None:
                        return None, None

                    try:
                        # sep=None with engine='python' attempts delimiter inference (comma/tab/etc)
                        tdf = pd.read_csv(fp, sep=None, engine="python")
                    except Exception:
                        return None, fp

                    # normalize columns (case-insensitive + strip whitespace)
                    tdf.columns = [str(c).strip() for c in tdf.columns]
                    cols_lc = {c.lower(): c for c in tdf.columns}

                    required = ["window", "sample", "pnl"]
                    if not all(r in cols_lc for r in required):
                        return None, fp

                    # rename required to canonical names
                    tdf = tdf.rename(columns={
                        cols_lc["window"]: "window",
                        cols_lc["sample"]: "sample",
                        cols_lc["pnl"]: "pnl",
                    })

                    # optional time columns for ordering
                    if "exit_time" in cols_lc:
                        tdf = tdf.rename(columns={cols_lc["exit_time"]: "exit_time"})
                    if "entry_time" in cols_lc:
                        tdf = tdf.rename(columns={cols_lc["entry_time"]: "entry_time"})

                    # normalize types
                    tdf["window"] = tdf["window"].astype(str).str.strip().map(normalize_window_label_to_active)
                    tdf["sample"] = tdf["sample"].astype(str).str.strip()
                    tdf["pnl"] = pd.to_numeric(tdf["pnl"], errors="coerce")

                    # drop rows with bad pnl
                    tdf = tdf.dropna(subset=["pnl"])

                    return tdf, fp

                def make_aggregate_summary_row(meta: dict, tdf: pd.DataFrame, mode: str):
                    """
                    mode:
                    - 'EXCL_LIVE' : W01 IS, W01 OOS, W02..W(HIST_WINDOWS) OOS
                    - 'ALL'       : W01 IS, W01 OOS, W02..W(EXPECTED_WINDOWS) OOS
                    - 'LIVE_ONLY' : W(LIVE_WA) OOS, W(LIVE_WB) OOS
                    - 'LIVE_MODE_HISTORY': W{LIVE_MODE_START} IS+OOS, then OOS through W{EXPECTED_WINDOWS}
                    """
                    if LIVE_MODE and mode == "LIVE_ONLY":
                        return None
                    sl = meta["sl"]
                    if sl is None or sl == 0:
                        return None

                    denom = 25.0 * sl

                    def sort_block(df):
                        if df is None or df.empty:
                            return df
                        if "exit_time" in df.columns:
                            return df.sort_values("exit_time")
                        if "entry_time" in df.columns:
                            return df.sort_values("entry_time")
                        return df

                    blocks = []

                    if mode in ("EXCL_LIVE", "ALL"):
                        # W01 IS
                        blocks.append(sort_block(
                            tdf[(tdf["window"] == "W01") & (tdf["sample"].str.upper() == "IS")]
                        ))
                        # W01 OOS
                        blocks.append(sort_block(
                            tdf[(tdf["window"] == "W01") & (tdf["sample"].str.upper() == "OOS")]
                        ))

                        last_w = HIST_WINDOWS if mode == "EXCL_LIVE" else EXPECTED_WINDOWS
                        for w in range(2, last_w + 1):
                            ww = f"W{w:02d}"
                            blocks.append(sort_block(
                                tdf[(tdf["window"] == ww) & (tdf["sample"].str.upper() == "OOS")]
                            ))

                    elif mode == "LIVE_ONLY":
                        for w in (LIVE_WA, LIVE_WB):
                            ww = f"W{w:02d}"
                            blocks.append(sort_block(
                                tdf[(tdf["window"] == ww) & (tdf["sample"].str.upper() == "OOS")]
                            ))
                    elif mode == "LIVE_MODE_HISTORY":
                        start_w = LIVE_MODE_START
                        if start_w > EXPECTED_WINDOWS:
                            return None
                        start_label = f"W{start_w:02d}"
                        blocks.append(sort_block(
                            tdf[(tdf["window"] == start_label) & (tdf["sample"].str.upper() == "IS")]
                        ))
                        blocks.append(sort_block(
                            tdf[(tdf["window"] == start_label) & (tdf["sample"].str.upper() == "OOS")]
                        ))
                        for w in range(start_w + 1, EXPECTED_WINDOWS + 1):
                            ww = f"W{w:02d}"
                            blocks.append(sort_block(
                                tdf[(tdf["window"] == ww) & (tdf["sample"].str.upper() == "OOS")]
                            ))

                    pnl = []
                    for b in blocks:
                        if b is None or b.empty:
                            continue
                        pnl_vals = pd.to_numeric(b["pnl"], errors="coerce").dropna().values
                        if pnl_vals.size:
                            pnl.append(pnl_vals)

                    if not pnl:
                        return None

                    pnl_all = np.concatenate(pnl)
                    r = pnl_all / denom
                    equity = np.cumsum(r)

                    peak = np.maximum.accumulate(equity)
                    dd = peak - equity

                    gross_profit = pnl_all[pnl_all > 0].sum()
                    gross_loss = pnl_all[pnl_all < 0].sum()

                    pf = np.inf if gross_loss == 0 and gross_profit > 0 else (
                        0.0 if gross_loss == 0 else gross_profit / abs(gross_loss)
                    )

                    return {
                        "StrategyFile": meta["strategy_file"],
                        "StopLoss_SL": sl,
                        "Trades": int(len(r)),
                        "ROI_R": float(np.sum(r)),
                        "PF": float(pf),
                        "EdgeLB": np.nan,
                        "MaxDD_R": float(np.max(dd)) if dd.size else 0.0,
                    }

                def sort_block_by_time(df_block: pd.DataFrame):
                    if df_block is None or df_block.empty:
                        return df_block
                    if "exit_time" in df_block.columns:
                        return df_block.sort_values("exit_time")
                    if "entry_time" in df_block.columns:
                        return df_block.sort_values("entry_time")
                    return df_block

                def _hist_window_range():
                    if LIVE_MODE:
                        return LIVE_MODE_START, EXPECTED_WINDOWS
                    return 1, HIST_WINDOWS

                def build_history_equity_curve(meta: dict, tdf: pd.DataFrame):
                    """
                    Returns cumulative R curve using W01 IS, W01 OOS, and OOS for W02..W{HIST_WINDOWS}.
                    """
                    sl = meta["sl"]
                    if sl is None or sl == 0:
                        return None

                    denom = 25.0 * sl
                    start_w, end_w = _hist_window_range()
                    start_label = f"W{start_w:02d}"
                    block_specs = [(start_label, "IS"), (start_label, "OOS")] + [
                        (f"W{w:02d}", "OOS") for w in range(start_w + 1, end_w + 1)
                    ]

                    pnl_parts = []
                    for w_label, sample in block_specs:
                        b = tdf[
                            (tdf["window"].astype(str) == w_label)
                            & (tdf["sample"].astype(str).str.upper() == sample)
                        ]
                        b = sort_block_by_time(b)
                        pnl_vals = pd.to_numeric(b["pnl"], errors="coerce").dropna().values.astype(float)
                        if pnl_vals.size:
                            pnl_parts.append(pnl_vals / denom)

                    if not pnl_parts:
                        return None

                    r_all = np.concatenate(pnl_parts)
                    return np.cumsum(r_all)

                def build_live_equity_curve(meta: dict, tdf: pd.DataFrame):
                    """Returns cumulative R curve using live proxy windows only."""
                    if LIVE_MODE:
                        return None
                    sl = meta["sl"]
                    if sl is None or sl == 0:
                        return None

                    denom = 25.0 * sl
                    block_specs = [(f"W{LIVE_WA:02d}", "OOS"), (f"W{LIVE_WB:02d}", "OOS")]

                    pnl_parts = []
                    for w_label, sample in block_specs:
                        b = tdf[
                            (tdf["window"].astype(str) == w_label)
                            & (tdf["sample"].astype(str).str.upper() == sample)
                        ]
                        b = sort_block_by_time(b)
                        pnl_vals = pd.to_numeric(b["pnl"], errors="coerce").dropna().values.astype(float)
                        if pnl_vals.size:
                            pnl_parts.append(pnl_vals / denom)

                    if not pnl_parts:
                        return None

                    r_all = np.concatenate(pnl_parts)
                    return np.cumsum(r_all)

                def _portfolio_sort_key(df_block):
                    if df_block is None or df_block.empty:
                        return pd.Series(dtype=object)
                    if "exit_time" in df_block.columns:
                        return df_block["exit_time"]
                    if "entry_time" in df_block.columns:
                        return df_block["entry_time"]
                    return df_block.index

                def _filter_trades_for_mode(tdf: pd.DataFrame, mode: str):
                    """
                    mode: 'HIST' or 'LIVE'
                    HIST: W01 IS, W01 OOS, W02..W{HIST_WINDOWS} OOS
                    LIVE: W{LIVE_WA} OOS, W{LIVE_WB} OOS
                    """
                    if mode == "HIST":
                        start_w, end_w = _hist_window_range()
                        windows_keep = [f"W{w:02d}" for w in range(start_w, end_w + 1)]
                        start_label = f"W{start_w:02d}"
                        samples = {start_label: {"IS", "OOS"}}
                        for w in range(start_w + 1, end_w + 1):
                            ww = f"W{w:02d}"
                            samples[ww] = {"OOS"}
                    else:
                        if LIVE_MODE:
                            return tdf.iloc[0:0].copy()
                        windows_keep = [f"W{LIVE_WA:02d}", f"W{LIVE_WB:02d}"]
                        samples = {w: {"OOS"} for w in windows_keep}

                    mask_rows = []
                    for idx, row in tdf.iterrows():
                        w = str(row["window"]).strip()
                        s = str(row["sample"]).strip().upper()
                        if w in windows_keep and s in samples.get(w, set()):
                            mask_rows.append(idx)
                    return tdf.loc[mask_rows].copy()

                def _portfolio_metrics_from_trades(trades_r: np.ndarray):
                    if trades_r.size == 0:
                        return {"ROI_R": np.nan, "PF": np.nan, "EdgeLB": np.nan, "Trades": 0, "Equity": np.array([], dtype=float), "MaxDD_R": np.nan}
                    equity = np.cumsum(trades_r)
                    peak = np.maximum.accumulate(equity)
                    dd = peak - equity
                    gross_profit = trades_r[trades_r > 0].sum()
                    gross_loss = trades_r[trades_r < 0].sum()
                    pf = np.inf if gross_loss == 0 and gross_profit > 0 else (0.0 if gross_loss == 0 else gross_profit / abs(gross_loss))
                    return {
                        "ROI_R": float(np.sum(trades_r)),
                        "PF": float(pf),
                        "EdgeLB": np.nan,
                        "Trades": int(trades_r.size),
                        "Equity": equity,
                        "MaxDD_R": float(np.max(dd)) if dd.size else np.nan,
                    }

                def _compute_sharpe(r_vals: np.ndarray):
                    if r_vals is None or r_vals.size < 2:
                        return np.nan
                    std = np.std(r_vals, ddof=1)
                    if std <= 0:
                        return np.nan
                    return float(np.mean(r_vals) / std * np.sqrt(r_vals.size))

                def _make_sorted_trades_from_info(info: dict, mode: str):
                    tdf_local = info["tdf"]
                    sl_local = info["sl"]
                    if sl_local is None or sl_local == 0:
                        return pd.DataFrame(columns=["sort_key", "r_unit"])
                    tr = _filter_trades_for_mode(tdf_local, mode)
                    if tr is None or tr.empty:
                        return pd.DataFrame(columns=["sort_key", "r_unit"])
                    tr = tr.copy()
                    tr["sort_key"] = _portfolio_sort_key(tr)
                    tr["r_unit"] = pd.to_numeric(tr["pnl"], errors="coerce") / (25.0 * sl_local)
                    tr = tr[["sort_key", "r_unit"]].dropna(subset=["r_unit"])
                    return tr.sort_values("sort_key")

                def _make_sorted_trades_for_windows(info: dict, window_nums):
                    if LIVE_MODE:
                        return pd.DataFrame(columns=["sort_key", "r_unit"])
                    tdf_local = info["tdf"]
                    sl_local = info["sl"]
                    if sl_local is None or sl_local == 0:
                        return pd.DataFrame(columns=["sort_key", "r_unit"])
                    windows_keep = {f"W{w:02d}" for w in window_nums}
                    mask = tdf_local["window"].astype(str).isin(windows_keep) & (tdf_local["sample"].astype(str).str.upper() == "OOS")
                    tr = tdf_local.loc[mask].copy()
                    if tr is None or tr.empty:
                        return pd.DataFrame(columns=["sort_key", "r_unit"])
                    tr["sort_key"] = _portfolio_sort_key(tr)
                    tr["r_unit"] = pd.to_numeric(tr["pnl"], errors="coerce") / (25.0 * sl_local)
                    tr = tr[["sort_key", "r_unit"]].dropna(subset=["r_unit"])
                    return tr.sort_values("sort_key")

                def _merge_sorted_trades(keys_list, r_list):
                    if not keys_list:
                        return np.array([], dtype=float), np.array([], dtype=float)
                    if all(len(r) == 0 for r in r_list):
                        return np.array([], dtype=float), np.array([], dtype=float)
                    keys = np.concatenate([k for k in keys_list if len(k) > 0])
                    r = np.concatenate([rv for rv in r_list if len(rv) > 0])
                    if r.size == 0:
                        return np.array([], dtype=float), np.array([], dtype=float)
                    order = np.argsort(keys, kind="mergesort")
                    return keys[order], r[order]

                def _build_portfolio_trades_multi(info_list, mode: str):
                    """
                    Builds combined trades (equal weight) for an arbitrary number of strategies.
                    mode: 'HIST' or 'LIVE'
                    """
                    if not info_list:
                        return np.array([], dtype=float)
                    n = len(info_list)
                    weight = 1.0 / n
                    parts = []
                    for info in info_list:
                        tr_sorted = info.get("sorted_hist") if mode == "HIST" else info.get("sorted_live")
                        if tr_sorted is None or tr_sorted.empty:
                            continue
                        part = tr_sorted.copy()
                        part["r_w"] = part["r_unit"] * weight
                        parts.append(part[["sort_key", "r_w"]])
                    if not parts:
                        return np.array([], dtype=float)
                    combined = pd.concat(parts, ignore_index=True)
                    combined = combined.sort_values("sort_key")
                    r_vals = pd.to_numeric(combined["r_w"], errors="coerce").dropna().values
                    return r_vals

                def _build_portfolio_trades_multi_df(info_list, mode: str):
                    """
                    Same as _build_portfolio_trades_multi but returns a sorted DataFrame with sort_key and r_w.
                    """
                    if not info_list:
                        return pd.DataFrame(columns=["sort_key", "r_w"])
                    n = len(info_list)
                    weight = 1.0 / n
                    parts = []
                    for info in info_list:
                        sl_local = info["sl"]
                        if sl_local is None or sl_local == 0:
                            continue
                        tdf_local = info["tdf"]
                        if mode == "HIST":
                            tr = _filter_trades_for_mode(tdf_local, "HIST")
                        else:
                            tr = _filter_trades_for_mode(tdf_local, "LIVE")
                        if tr is None or tr.empty:
                            continue
                        tr = tr.copy()
                        tr["sort_key"] = _portfolio_sort_key(tr)
                        tr["r_w"] = pd.to_numeric(tr["pnl"], errors="coerce") / (25.0 * sl_local) * weight
                        tr = tr[["sort_key", "r_w"]].dropna(subset=["r_w"])
                        if not tr.empty:
                            parts.append(tr)
                    if not parts:
                        return pd.DataFrame(columns=["sort_key", "r_w"])
                    combined = pd.concat(parts, ignore_index=True)
                    combined = combined.sort_values("sort_key")
                    return combined

                def _build_portfolio_trades_multi_for_windows(info_list, window_nums):
                    """
                    Equal-weight trades restricted to specific OOS window numbers (e.g., [LIVE_WA]).
                    """
                    if LIVE_MODE:
                        return np.array([], dtype=float)
                    if not info_list:
                        return np.array([], dtype=float)
                    windows_keep = {f"W{w:02d}" for w in window_nums}
                    n = len(info_list)
                    weight = 1.0 / n
                    parts = []
                    for info in info_list:
                        sl_local = info["sl"]
                        if sl_local is None or sl_local == 0:
                            continue
                        tdf_local = info["tdf"]
                        mask = tdf_local["window"].astype(str).isin(windows_keep) & (tdf_local["sample"].astype(str).str.upper() == "OOS")
                        tr = tdf_local.loc[mask].copy()
                        if tr.empty:
                            continue
                        tr["sort_key"] = _portfolio_sort_key(tr)
                        tr["r_w"] = pd.to_numeric(tr["pnl"], errors="coerce") / (25.0 * sl_local) * weight
                        tr = tr[["sort_key", "r_w"]].dropna(subset=["r_w"])
                        if not tr.empty:
                            parts.append(tr)
                    if not parts:
                        return np.array([], dtype=float)
                    combined = pd.concat(parts, ignore_index=True)
                    combined = combined.sort_values("sort_key")
                    r_vals = pd.to_numeric(combined["r_w"], errors="coerce").dropna().values
                    return r_vals

                # Build three summary sheets:
                # 1) Excluding live proxies: W01 IS, W01 OOS, and OOS for W02..W{HIST_WINDOWS}
                # 2) Including live proxies: W01 IS, W01 OOS, and OOS for W02..W{EXPECTED_WINDOWS}
                # 3) Only live proxies: OOS for W{LIVE_WA} and W{LIVE_WB}
                sheet_excl = []
                sheet_all = []
                sheet_live = []

                equity_curves = []
                hist_equity_curves = []
                hist_heatmap_points = []
                strategies_data = {}  # strategy_file -> dict with metrics/curves/trade data
                allowed_strategies = set()
                allowed_strategies_live = set()
                live_mode_rows = []

                for meta in passing_meta_all:
                    tdf, _ = load_trade_list(meta["strategy_dir"])
                    if tdf is None or tdf.empty:
                        continue

                    if LIVE_MODE:
                        row_excl = make_aggregate_summary_row(meta, tdf, "LIVE_MODE_HISTORY")
                        row_all = row_excl
                        row_live = None
                    else:
                        row_excl = make_aggregate_summary_row(meta, tdf, "EXCL_LIVE")
                        row_all  = make_aggregate_summary_row(meta, tdf, "ALL")
                        row_live = make_aggregate_summary_row(meta, tdf, "LIVE_ONLY")
                    row_live_mode_hist = make_aggregate_summary_row(meta, tdf, "LIVE_MODE_HISTORY")
                    hist_eq = None
                    live_eq = None

                    # cache per-strategy data for portfolios
                    strategies_data[meta["strategy_file"]] = {
                        "meta": meta,
                        "row_excl": row_excl,
                        "row_live": row_live,
                        "row_live_mode": row_live_mode_hist,
                        "hist_eq": hist_eq,
                        "live_eq": live_eq,
                        "tdf": tdf.copy(),
                        "sl": meta["sl"],
                    }

                    # Standard pipeline exports/gates
                    if meta.get("passed_standard"):
                        if row_excl and row_excl.get("Trades", 0) >= 100:
                            pf_hist = row_excl.get("PF", np.nan)
                            if np.isfinite(pf_hist) and pf_hist >= PF_OK:
                                allowed_strategies.add(meta["strategy_file"])

                                sheet_excl.append(row_excl)
                                if row_all:
                                    sheet_all.append(row_all)
                                if row_live and not LIVE_MODE:
                                    sheet_live.append(row_live)

                                if row_excl and row_live and not LIVE_MODE:
                                    if np.isfinite(row_excl["PF"]) and np.isfinite(row_excl["EdgeLB"]) and np.isfinite(row_live["PF"]):
                                        hist_heatmap_points.append((row_excl["PF"], row_excl["EdgeLB"], row_live["PF"]))

                                hist_eq = build_history_equity_curve(meta, tdf)
                                if hist_eq is not None and hist_eq.size:
                                    hist_equity_curves.append((meta["strategy_file"], hist_eq))

                                live_eq = build_live_equity_curve(meta, tdf)
                                strategies_data[meta["strategy_file"]]["hist_eq"] = hist_eq
                                strategies_data[meta["strategy_file"]]["live_eq"] = live_eq

                                # equity curve for plotting (ALL windows) - built from W01 IS -> W01 OOS -> W02..Wn OOS
                                sl_local = meta["sl"]
                                denom = 25.0 * sl_local if (sl_local is not None and sl_local != 0) else None
                                if denom:
                                    def sort_block(df_block):
                                        if df_block is None or df_block.empty:
                                            return df_block
                                        if "exit_time" in df_block.columns:
                                            return df_block.sort_values("exit_time")
                                        if "entry_time" in df_block.columns:
                                            return df_block.sort_values("entry_time")
                                        return df_block

                                    blocks = []
                                    b = sort_block(tdf[(tdf["window"].astype(str) == "W01") & (tdf["sample"].astype(str).str.upper() == "IS")])
                                    blocks.append(b)
                                    b = sort_block(tdf[(tdf["window"].astype(str) == "W01") & (tdf["sample"].astype(str).str.upper() == "OOS")])
                                    blocks.append(b)
                                    for w in range(2, EXPECTED_WINDOWS + 1):
                                        ww = f"W{w:02d}"
                                        b = sort_block(tdf[(tdf["window"].astype(str) == ww) & (tdf["sample"].astype(str).str.upper() == "OOS")])
                                        blocks.append(b)

                                    pnl_concat = []
                                    for bb in blocks:
                                        if bb is None or bb.empty:
                                            continue
                                        pnl_vals = pd.to_numeric(bb["pnl"], errors="coerce").dropna().values.astype(float)
                                        if pnl_vals.size:
                                            pnl_concat.append(pnl_vals)

                                    if pnl_concat:
                                        pnl_all = np.concatenate(pnl_concat)
                                        r_all = pnl_all / denom
                                        eq = np.cumsum(r_all)
                                        equity_curves.append((meta["strategy_file"], eq))

                    # Live-mode pipeline gating (skip first two windows)
                    if meta.get("passed_live_mode") and row_live_mode_hist:
                        pf_lm = row_live_mode_hist.get("PF", np.nan)
                        trades_lm = row_live_mode_hist.get("Trades", 0)
                        if np.isfinite(pf_lm) and pf_lm > 1.0 and trades_lm >= 100:
                            allowed_strategies_live.add(meta["strategy_file"])

                    # Precompute sorted trade lists for portfolio construction
                for sname, sinfo in strategies_data.items():
                    sinfo["sorted_hist"] = _make_sorted_trades_from_info(sinfo, "HIST")
                    sinfo["sorted_live"] = _make_sorted_trades_from_info(sinfo, "LIVE")
                    sinfo["sorted_live_wA"] = _make_sorted_trades_for_windows(sinfo, [LIVE_WA])
                    sinfo["sorted_live_wB"] = _make_sorted_trades_for_windows(sinfo, [LIVE_WB])
                    sinfo["hist_keys"] = sinfo["sorted_hist"]["sort_key"].to_numpy()
                    sinfo["hist_r"] = sinfo["sorted_hist"]["r_unit"].to_numpy(dtype=float)
                    sinfo["live_keys"] = sinfo["sorted_live"]["sort_key"].to_numpy()
                    sinfo["live_r"] = sinfo["sorted_live"]["r_unit"].to_numpy(dtype=float)
                    sinfo["live_wA_keys"] = sinfo["sorted_live_wA"]["sort_key"].to_numpy()
                    sinfo["live_wA_r"] = sinfo["sorted_live_wA"]["r_unit"].to_numpy(dtype=float)
                    sinfo["live_wB_keys"] = sinfo["sorted_live_wB"]["sort_key"].to_numpy()
                    sinfo["live_wB_r"] = sinfo["sorted_live_wB"]["r_unit"].to_numpy(dtype=float)

                # -------------------------------
                # Build equal-weight portfolios (random from top-N by selected metric; size = PORTFOLIO_SIZE)
                # -------------------------------
                portfolio5_rows = []
                portfolio_live_curves = []  # keep live equity for plotting/statistics
                if strategies_data:
                    eligible_rank = []
                    for name, info in strategies_data.items():
                        if name not in allowed_strategies:
                            continue
                        row_excl = info.get("row_excl")
                        if row_excl is None:
                            continue
                        trades_hist = row_excl.get("Trades", 0)
                        hist_r = info.get("hist_r", np.array([], dtype=float))
                        if trades_hist < 100:
                            continue

                        metric = TOP_STRATEGY_RANK_METRIC.strip().lower()
                        if metric == "pf":
                            pf_hist = row_excl.get("PF", np.nan)
                            if np.isfinite(pf_hist):
                                eligible_rank.append((name, float(pf_hist)))
                        elif metric == "maxdd":
                            if hist_r is not None and hist_r.size >= 2:
                                eq = np.cumsum(hist_r)
                                peak = np.maximum.accumulate(eq)
                                dd = peak - eq
                                maxdd = float(np.max(dd)) if dd.size else np.nan
                                if np.isfinite(maxdd):
                                    # lower is better, so store negative to sort descending
                                    eligible_rank.append((name, -maxdd))
                        else:
                            if hist_r is not None and hist_r.size >= 2:
                                sh = _compute_sharpe(hist_r)
                                if np.isfinite(sh):
                                    eligible_rank.append((name, float(sh)))

                    # sort by selected metric descending and keep top-N
                    eligible_rank.sort(key=lambda x: x[1], reverse=True)
                    top_names_all = [name for name, _ in eligible_rank[:TOP_STRATEGIES_POOL]]

                    # Randomly sample unique portfolios (faster than generating lexicographic combinations)
                    if len(top_names_all) >= PORTFOLIO_SIZE:
                        rng = np.random.default_rng(42)
                        max_portfolios = MAX_PORTFOLIO_COMBOS
                        max_unique = math.comb(len(top_names_all), PORTFOLIO_SIZE)
                        if max_unique < max_portfolios:
                            log_line(
                                f"[Portfolios] Unique combo limit is {max_unique} "
                                f"(n={len(top_names_all)}, size={PORTFOLIO_SIZE}); cannot reach {max_portfolios} unique portfolios."
                            )
                        combos_idx = set()
                        attempts = 0
                        max_attempts = max_portfolios * 50
                        n_names = len(top_names_all)
                        while len(combos_idx) < max_portfolios and attempts < max_attempts:
                            pick = tuple(sorted(rng.choice(n_names, size=PORTFOLIO_SIZE, replace=False)))
                            combos_idx.add(pick)
                            attempts += 1

                        combos_list = [tuple(top_names_all[i] for i in idxs) for idxs in sorted(combos_idx)]
                    else:
                        combos_list = []

                    for combo in combos_list:
                        infos = [strategies_data[n] for n in combo]
                        n_infos = len(infos)
                        weight = 1.0 / n_infos if n_infos else 0.0

                        # HIST (array-based merge)
                        hist_keys, hist_r = _merge_sorted_trades(
                            [i["hist_keys"] for i in infos],
                            [i["hist_r"] * weight for i in infos]
                        )

                        # LIVE (array-based merge)
                        live_keys, live_r = _merge_sorted_trades(
                            [i["live_keys"] for i in infos],
                            [i["live_r"] * weight for i in infos]
                        )

                        hist_metrics = _portfolio_metrics_from_trades(hist_r)
                        live_metrics = _portfolio_metrics_from_trades(live_r)

                        # LIVE window-specific (array-based merge)
                        _, live_wA_r = _merge_sorted_trades(
                            [i["live_wA_keys"] for i in infos],
                            [i["live_wA_r"] * weight for i in infos]
                        )
                        _, live_wB_r = _merge_sorted_trades(
                            [i["live_wB_keys"] for i in infos],
                            [i["live_wB_r"] * weight for i in infos]
                        )
                        live_wA_metrics = _portfolio_metrics_from_trades(live_wA_r)
                        live_wB_metrics = _portfolio_metrics_from_trades(live_wB_r)

                        # keep live equity curve for top-30 plotting
                        live_eq = np.cumsum(live_r) if live_r.size else np.array([], dtype=float)
                        portfolio_live_curves.append({
                            "Strategies": " + ".join(combo),
                            "Hist_PF": hist_metrics["PF"],
                            "Live_EQ": live_eq,
                            "Live_Sort_Keys": live_keys.tolist() if live_keys.size else [],
                            "Live_R": live_r,
                            "Live_WA_R": live_wA_r,
                            "Live_WB_R": live_wB_r,
                        })

                        portfolio5_rows.append({
                            "Strategies": " + ".join(combo),
                            "Hist_PF": hist_metrics["PF"],
                            "Hist_ROI_R": hist_metrics["ROI_R"],
                            "Hist_Trades": hist_metrics["Trades"],
                            "Hist_MaxDD_R": hist_metrics["MaxDD_R"],
                            "Hist_Sharpe": _compute_sharpe(hist_r),
                            "Live_PF": live_metrics["PF"],
                            "Live_ROI_R": live_metrics["ROI_R"],
                            "Live_Trades": live_metrics["Trades"],
                            "Live_MaxDD_R": live_metrics["MaxDD_R"],
                            "Live_Sharpe": _compute_sharpe(live_r),
                            f"Live_W{LIVE_WA}_PF": live_wA_metrics["PF"],
                            f"Live_W{LIVE_WA}_ROI_R": live_wA_metrics["ROI_R"],
                            f"Live_W{LIVE_WA}_Trades": live_wA_metrics["Trades"],
                            f"Live_W{LIVE_WA}_MaxDD_R": live_wA_metrics["MaxDD_R"],
                            f"Live_W{LIVE_WA}_Sharpe": _compute_sharpe(live_wA_r),
                            f"Live_W{LIVE_WB}_PF": live_wB_metrics["PF"],
                            f"Live_W{LIVE_WB}_ROI_R": live_wB_metrics["ROI_R"],
                            f"Live_W{LIVE_WB}_Trades": live_wB_metrics["Trades"],
                            f"Live_W{LIVE_WB}_MaxDD_R": live_wB_metrics["MaxDD_R"],
                            f"Live_W{LIVE_WB}_Sharpe": _compute_sharpe(live_wB_r),
                        })

                # Build main export DataFrame after we know which strategies cleared the trade-list 100+ history gate
                df = pd.DataFrame([r for r in passing_rows if r.get("StrategyFile") in allowed_strategies])
                front = ["StrategyFile", "FullPath", "StopLoss_SL", "TotalTrades_AllBaseSamples"]
                cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
                df = df[cols]

                df_excl = pd.DataFrame(sheet_excl) if sheet_excl else pd.DataFrame()
                df_all2 = pd.DataFrame(sheet_all) if sheet_all else pd.DataFrame()
                df_live = pd.DataFrame(sheet_live) if sheet_live else pd.DataFrame()
                df_portfolios = pd.DataFrame(portfolio5_rows) if portfolio5_rows else pd.DataFrame()
                _backtest_port_roi_out = os.environ.get(ENV_BACKTEST_PORT_ROI_OUT, "").strip()
                _backtest_skip_xlsx = os.environ.get(ENV_BACKTEST_SKIP_XLSX, "") == "1"

                if _backtest_port_roi_out:
                    live_roi_vals = [
                        float(p.get("Live_ROI_R"))
                        for p in portfolio5_rows
                        if np.isfinite(p.get("Live_ROI_R", np.nan))
                    ]
                    payload = {
                        "select_test_id": SELECT_TEST_ID,
                        "windows_offset": WINDOW_OFFSET,
                        "portfolio_count": len(live_roi_vals),
                        "live_roi_r": live_roi_vals,
                    }
                    with open(_backtest_port_roi_out, "w", encoding="utf-8") as fout:
                        json.dump(payload, fout)
                    log_line(
                        f"[Backtest child] Saved portfolio live ROI values to {_backtest_port_roi_out} "
                        f"(n={len(live_roi_vals)})."
                    )
                    if _backtest_skip_xlsx:
                        raise SystemExit(0)

                if LIVE_MODE:
                    out_path = os.path.join(root_dir, f"selected_pipeline_{SELECT_TEST_ID}_strategies_live.xlsx")
                else:
                    out_path = os.path.join(root_dir, f"selected_pipeline_{SELECT_TEST_ID}_strategies.xlsx")

                # Write ALL sheets into the same workbook
                with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="All_Parsed_From_TXT", index=False)

                    # New sheets requested
                    df_excl.to_excel(writer, sheet_name="TradeList_ExclLive", index=False)
                    df_all2.to_excel(writer, sheet_name="TradeList_AllWindows", index=False)
                    df_live.to_excel(writer, sheet_name="TradeList_LiveOnly", index=False)

                    # Portfolios (equal weight, top history performers by TOP_STRATEGY_RANK_METRIC)
                    if not df_portfolios.empty:
                        df_portfolios.to_excel(writer, sheet_name="Portfolios", index=False)
                    else:
                        pd.DataFrame().to_excel(writer, sheet_name="Portfolios", index=False)

                log_line("\n==================== EXPORT COMPLETE ====================\n")
                log_line(f"Selected test id: {SELECT_TEST_ID}")
                log_line(f"Metric reported: {BEST_METRIC}")
                log_line(f"Lower Bound (%): {selected_row['lower_bound']:.2f}")
                log_line(f"Point Estimate (%): {selected_row['point_est']:.2f}")
                log_line(f"n: {selected_row['n']}")
                log_line(f"Excel saved to: {out_path}")
                log_line(f"Rows exported (All_Parsed_From_TXT): {len(df)}")
                log_line(f"Rows exported (TradeList_ExclLive): {len(df_excl)}")
                log_line(f"Rows exported (TradeList_AllWindows): {len(df_all2)}")
                log_line(f"Rows exported (TradeList_LiveOnly): {len(df_live)}")
                log_line(f"Portfolios ({PORTFOLIO_SIZE}-strategy equal-weight): {len(df_portfolios)}")

                # -------------------------------
                # Portfolio-level live stats and plotting
                # -------------------------------
                def _first_index_ge(eq: np.ndarray, target: float):
                    hits = np.where(eq >= target)[0]
                    return int(hits[0]) if hits.size else None

                def _first_index_le(eq: np.ndarray, threshold: float):
                    hits = np.where(eq <= threshold)[0]
                    return int(hits[0]) if hits.size else None

                def _day_loss_breach(sort_keys, r_vals, limit=-4.0):
                    """
                    Returns index of first trade where same-day cumulative sum <= limit.
                    If timestamps are missing, returns None (assumes no breach).
                    """
                    if not sort_keys or len(sort_keys) != len(r_vals):
                        return None
                    ts = pd.to_datetime(sort_keys, errors="coerce")
                    if ts.isna().all():
                        return None
                    df_tmp = pd.DataFrame({"ts": ts, "r": r_vals})
                    df_tmp["day"] = df_tmp["ts"].dt.date
                    day_sum = 0.0
                    current_day = None
                    first_idx = None
                    for idx, row in df_tmp.iterrows():
                        d = row["day"]
                        if current_day is None or d != current_day:
                            current_day = d
                            day_sum = 0.0
                        day_sum += row["r"]
                        if day_sum <= limit:
                            first_idx = idx
                            break
                    return int(first_idx) if first_idx is not None else None

                def _trade_day_candle_to_target(eq: np.ndarray, sort_keys, target: float):
                    """
                    Returns (trades_to_target, days_to_target, candles_to_target) if hit, else (None, None, None).
                    days are calendar-day span based on sort_key timestamps (if available).
                    candles is trade count (proxy).
                    """
                    idx = _first_index_ge(eq, target)
                    if idx is None:
                        return None, None, None
                    candles = idx + 1  # trade count proxy
                    days = None
                    if sort_keys and len(sort_keys) > idx:
                        ts = pd.Series(pd.to_datetime(sort_keys, errors="coerce"))
                        ts_valid = ts.dropna()
                        if not ts_valid.empty and pd.notna(ts.iloc[idx]):
                            t0 = ts_valid.iloc[0]
                            t_hit = ts.iloc[idx]
                            days = (t_hit - t0).total_seconds() / 86400.0
                    return candles, days, candles

                log_line("\n==================== PORTFOLIO LIVE STATS ====================\n")
                log_line(f"Total portfolios: {len(portfolio5_rows)}")

                if USE_PCT_MODE:
                    def _equity_mult_from_r(r_vals: np.ndarray) -> np.ndarray:
                        if r_vals is None or r_vals.size == 0:
                            return np.array([], dtype=float)
                        return np.cumprod(1.0 + (r_vals * RISK_PCT_PER_R))

                    equity_mults = []
                    equity_mult_pairs = []
                    for pcurve in portfolio_live_curves:
                        eqm = _equity_mult_from_r(pcurve.get("Live_R", np.array([], dtype=float)))
                        if eqm.size:
                            equity_mults.append(eqm)
                            equity_mult_pairs.append((pcurve, eqm))

                    final_mults = [eqm[-1] for eqm in equity_mults if eqm.size]
                    prof_mask = [v > 1.0 for v in final_mults]
                    profitable_count = sum(prof_mask)
                    unprofitable_count = len(final_mults) - profitable_count

                    log_line(f"Profitable (final equity > 1.00x): {profitable_count}")
                    log_line(f"Unprofitable (final equity <= 1.00x): {unprofitable_count}")

                    # Average W5/W6 (live proxies) results in percentage and average max DD (percent)
                    wA_mults = []
                    wB_mults = []
                    maxdd_pcts = []
                    for pcurve in portfolio_live_curves:
                        eqm = _equity_mult_from_r(pcurve.get("Live_R", np.array([], dtype=float)))
                        if eqm.size:
                            peak = np.maximum.accumulate(eqm)
                            dd = 1.0 - (eqm / peak)
                            if dd.size:
                                maxdd_pcts.append(float(np.max(dd)) * 100.0)
                        eqm_wA = _equity_mult_from_r(pcurve.get("Live_WA_R", np.array([], dtype=float)))
                        if eqm_wA.size:
                            wA_mults.append(eqm_wA[-1])
                        eqm_wB = _equity_mult_from_r(pcurve.get("Live_WB_R", np.array([], dtype=float)))
                        if eqm_wB.size:
                            wB_mults.append(eqm_wB[-1])

                    avg_wA_pct = (np.mean(wA_mults) - 1.0) * 100.0 if wA_mults else np.nan
                    avg_wB_pct = (np.mean(wB_mults) - 1.0) * 100.0 if wB_mults else np.nan
                    avg_live_pct = (np.mean(final_mults) - 1.0) * 100.0 if final_mults else np.nan
                    avg_maxdd_pct = float(np.mean(maxdd_pcts)) if maxdd_pcts else np.nan

                    log_line(f"Average W{LIVE_WA}&W{LIVE_WB} result (%): {avg_live_pct:.2f}%")
                    log_line(f"Average W{LIVE_WA} result (%): {avg_wA_pct:.2f}%")
                    log_line(f"Average W{LIVE_WB} result (%): {avg_wB_pct:.2f}%")
                    log_line(f"Average max drawdown (%): {avg_maxdd_pct:.2f}%")

                    # Milestones and drawdown cliffs
                    milestones = [
                        ("Reached +25% (1.25x)", 1.25),
                        ("Reached +50% (1.50x)", 1.50),
                        ("Reached +100% (2.00x)", 2.00),
                        ("Reached +200% (3.00x)", 3.00),
                    ]
                    cliffs = [
                        ("Reached -10% (0.90x)", 0.90),
                        ("Reached -20% (0.80x)", 0.80),
                        ("Reached -30% (0.70x)", 0.70),
                        ("Reached -50% (0.50x)", 0.50),
                    ]

                    def _first_idx_ge(eqm: np.ndarray, thr: float):
                        hits = np.where(eqm >= thr)[0]
                        return int(hits[0]) if hits.size else None

                    def _first_idx_le(eqm: np.ndarray, thr: float):
                        hits = np.where(eqm <= thr)[0]
                        return int(hits[0]) if hits.size else None

                    def _first_idx_ge_from(eqm: np.ndarray, thr: float, start_idx: int):
                        if eqm is None or eqm.size == 0 or start_idx >= eqm.size:
                            return None
                        hits = np.where(eqm[start_idx:] >= thr)[0]
                        return int(hits[0] + start_idx) if hits.size else None

                    def _first_idx_le_from(eqm: np.ndarray, thr: float, start_idx: int):
                        if eqm is None or eqm.size == 0 or start_idx >= eqm.size:
                            return None
                        hits = np.where(eqm[start_idx:] <= thr)[0]
                        return int(hits[0] + start_idx) if hits.size else None

                    def _day_loss_breach_pct(eqm: np.ndarray, sort_keys, limit: float = -0.05):
                        """
                        Returns index of first trade where same-day equity drawdown <= limit.
                        Uses day-start equity as baseline. If timestamps missing, returns None.
                        """
                        if eqm is None or eqm.size == 0 or not sort_keys or len(sort_keys) != len(eqm):
                            return None
                        ts = pd.to_datetime(sort_keys, errors="coerce")
                        if ts.isna().all():
                            return None
                        current_day = None
                        day_start = 1.0
                        for i, t in enumerate(ts):
                            if pd.isna(t):
                                continue
                            d = t.date()
                            if current_day is None or d != current_day:
                                current_day = d
                                day_start = eqm[i - 1] if i > 0 else 1.0
                            day_ret = (eqm[i] / day_start) - 1.0 if day_start != 0 else 0.0
                            if day_ret <= limit:
                                return int(i)
                        return None

                    def _day_loss_breach_pct_from(eqm: np.ndarray, sort_keys, limit: float, start_idx: int):
                        if eqm is None or eqm.size == 0 or start_idx >= eqm.size:
                            return None
                        if not sort_keys or len(sort_keys) != len(eqm):
                            return None
                        ts = pd.to_datetime(sort_keys, errors="coerce")
                        if ts.isna().all():
                            return None
                        current_day = None
                        day_start = 1.0
                        for i, t in enumerate(ts):
                            if i < start_idx:
                                continue
                            if pd.isna(t):
                                continue
                            d = t.date()
                            if current_day is None or d != current_day:
                                current_day = d
                                day_start = eqm[i - 1] if i > 0 else 1.0
                            day_ret = (eqm[i] / day_start) - 1.0 if day_start != 0 else 0.0
                            if day_ret <= limit:
                                return int(i)
                        return None

                    # Diagnostic test: count how many portfolios ever reach each upside milestone.
                    log_line("\n[Milestone Test] Portfolios reaching each equity growth milestone")
                    for label, thr in milestones:
                        count = 0
                        for eqm in equity_mults:
                            if _first_idx_ge(eqm, thr) is not None:
                                count += 1
                        log_line(f"  - {label}: {count}")

                    # Diagnostic test: count how many portfolios hit each adverse drawdown threshold.
                    log_line("\n[Drawdown Test] Portfolios reaching each drawdown cliff")
                    for label, thr in cliffs:
                        count = 0
                        for eqm in equity_mults:
                            if _first_idx_le(eqm, thr) is not None:
                                count += 1
                        log_line(f"  - {label}: {count}")

                    # First-passage ordering test: upside target must be reached before downside barrier.
                    log_line("\n[First-Passage Ordering Test] Upside target reached before downside barrier")
                    order_checks = [
                        ("Reached +25% before -20%", 1.25, 0.80),
                        ("Reached +50% before -30%", 1.50, 0.70),
                        ("Reached +100% before -50%", 2.00, 0.50),
                        ("Reached +10% before -10%", 1.10, 0.90),
                    ]
                    for label, up_thr, down_thr in order_checks:
                        count = 0
                        for eqm in equity_mults:
                            up_i = _first_idx_ge(eqm, up_thr)
                            dn_i = _first_idx_le(eqm, down_thr)
                            if up_i is not None and (dn_i is None or up_i < dn_i):
                                count += 1
                        log_line(f"  - {label}: {count}")

                    # Same ordering test, with an extra daily loss barrier.
                    log_line("\n[First-Passage + Day-Loss Test] +10% before -10% and before any -5% day")
                    count_10_before_10_and_day = 0
                    for pcurve, eqm in equity_mult_pairs:
                        up_i = _first_idx_ge(eqm, 1.10)
                        dn_i = _first_idx_le(eqm, 0.90)
                        day_i = _day_loss_breach_pct(eqm, pcurve.get("Live_Sort_Keys", []), limit=-0.05)
                        if up_i is not None and (dn_i is None or up_i < dn_i) and (day_i is None or up_i < day_i):
                            count_10_before_10_and_day += 1
                    log_line(f"  - Pass count: {count_10_before_10_and_day}")

                    # -------------------------------
                    # Challenge-style pass rates (compounding / % mode)
                    # -------------------------------
                    total_portfolios = len(equity_mult_pairs)
                    ss_count = 0
                    ts_step1_count = 0
                    ts_step2_count = 0
                    for pcurve, eqm in equity_mult_pairs:
                        keys = pcurve.get("Live_Sort_Keys", [])

                        # Single Step: +10% before -6% and before -4% day
                        up_i = _first_idx_ge(eqm, 1.10)
                        dn_i = _first_idx_le(eqm, 0.94)
                        day_i = _day_loss_breach_pct(eqm, keys, limit=-0.04)
                        if up_i is not None and (dn_i is None or up_i < dn_i) and (day_i is None or up_i < day_i):
                            ss_count += 1

                        # Two Step: Step 1 = +10% before -10% and before -5% day
                        up1_i = _first_idx_ge(eqm, 1.10)
                        dn1_i = _first_idx_le(eqm, 0.90)
                        day1_i = _day_loss_breach_pct(eqm, keys, limit=-0.05)
                        if up1_i is not None and (dn1_i is None or up1_i < dn1_i) and (day1_i is None or up1_i < day1_i):
                            ts_step1_count += 1

                            # Step 2 (after Step 1): +5% before -10% and before -5% day, relative to eq at step1
                            base = eqm[up1_i]
                            if base > 0:
                                eqm_rel = eqm / base
                                up2_i = _first_idx_ge_from(eqm_rel, 1.05, up1_i + 1)
                                dn2_i = _first_idx_le_from(eqm_rel, 0.90, up1_i + 1)
                                day2_i = _day_loss_breach_pct_from(eqm, keys, limit=-0.05, start_idx=up1_i + 1)
                                if up2_i is not None and (dn2_i is None or up2_i < dn2_i) and (day2_i is None or up2_i < day2_i):
                                    ts_step2_count += 1

                    log_line("\n[Challenge Test - Percentage Mode] Single-step rule")
                    log_line("  - Requirement: hit +10% before -6% drawdown and before any -4% day.")
                    log_line(f"  - Passes: {ss_count} / {total_portfolios}")
                    log_line("\n[Challenge Test - Percentage Mode] Two-step rule")
                    log_line("  - Step 1: hit +10% before -10% drawdown and before any -5% day.")
                    log_line("  - Step 2: after Step 1, hit +5% before -10% drawdown and before any -5% day.")
                    log_line(f"  - Step 1 passes: {ts_step1_count} / {total_portfolios}")
                    log_line(f"  - Full two-step passes: {ts_step2_count} / {total_portfolios}")

                    # -------------------------------
                    # Single-step follow-up stats (compounding / pct)
                    # -------------------------------
                    ss_lost_count = 0
                    ss_no_loss_count = 0
                    sum_max_profit_pct = 0.0
                    sum_trades_to_loss = 0.0
                    for pcurve, eqm in equity_mult_pairs:
                        keys = pcurve.get("Live_Sort_Keys", [])
                        up_i = _first_idx_ge(eqm, 1.10)
                        dn_i = _first_idx_le(eqm, 0.94)
                        day_i = _day_loss_breach_pct(eqm, keys, limit=-0.04)
                        if up_i is None or (dn_i is not None and dn_i < up_i) or (day_i is not None and day_i < up_i):
                            continue

                        loss_dd_i = _first_idx_le_from(eqm, 0.90, up_i + 1)
                        loss_day_i = _day_loss_breach_pct_from(eqm, keys, limit=-0.05, start_idx=up_i + 1)
                        loss_candidates = [i for i in [loss_dd_i, loss_day_i] if i is not None]
                        if not loss_candidates:
                            ss_no_loss_count += 1
                            continue

                        loss_i = min(loss_candidates)
                        max_eq = float(np.max(eqm[:loss_i + 1])) if loss_i is not None else float(np.max(eqm))
                        sum_max_profit_pct += (max_eq - 1.0) * 100.0
                        sum_trades_to_loss += (loss_i + 1)
                        ss_lost_count += 1

                    log_line("\n[Challenge Follow-up - Percentage Mode] Single-step survivors")
                    log_line("  - Post-pass loss condition: 10% drawdown OR 5% daily drawdown.")
                    if ss_lost_count:
                        log_line(f"  - Avg max profit before loss (%): {sum_max_profit_pct / ss_lost_count:.2f}")
                        log_line(f"  - Avg trades before loss: {sum_trades_to_loss / ss_lost_count:.2f}")
                    else:
                        log_line("  - Avg max profit before loss (%): n/a")
                        log_line("  - Avg trades before loss: n/a")
                    log_line(f"  - Did not hit post-pass loss condition: {ss_no_loss_count}")

                    # -------------------------------
                    # Two-step follow-up stats (compounding / pct)
                    # -------------------------------
                    ts_lost_count = 0
                    ts_no_loss_count = 0
                    ts_sum_max_profit_pct = 0.0
                    ts_sum_trades_to_loss = 0.0
                    for pcurve, eqm in equity_mult_pairs:
                        keys = pcurve.get("Live_Sort_Keys", [])

                        # Step 1
                        up1_i = _first_idx_ge(eqm, 1.10)
                        dn1_i = _first_idx_le(eqm, 0.90)
                        day1_i = _day_loss_breach_pct(eqm, keys, limit=-0.05)
                        if up1_i is None or (dn1_i is not None and dn1_i < up1_i) or (day1_i is not None and day1_i < up1_i):
                            continue

                        # Step 2 (after step 1), relative to eq at step1
                        base = eqm[up1_i]
                        if base <= 0:
                            continue
                        eqm_rel = eqm / base
                        up2_i = _first_idx_ge_from(eqm_rel, 1.05, up1_i + 1)
                        dn2_i = _first_idx_le_from(eqm_rel, 0.90, up1_i + 1)
                        day2_i = _day_loss_breach_pct_from(eqm, keys, limit=-0.05, start_idx=up1_i + 1)
                        if up2_i is None or (dn2_i is not None and dn2_i < up2_i) or (day2_i is not None and day2_i < up2_i):
                            continue

                        # Loss after completing step 2 (absolute loss rules)
                        loss_dd_i = _first_idx_le_from(eqm, 0.90, up2_i + 1)
                        loss_day_i = _day_loss_breach_pct_from(eqm, keys, limit=-0.05, start_idx=up2_i + 1)
                        loss_candidates = [i for i in [loss_dd_i, loss_day_i] if i is not None]
                        if not loss_candidates:
                            ts_no_loss_count += 1
                            continue

                        loss_i = min(loss_candidates)
                        max_eq = float(np.max(eqm[:loss_i + 1]))
                        ts_sum_max_profit_pct += (max_eq - 1.0) * 100.0
                        ts_sum_trades_to_loss += (loss_i + 1)
                        ts_lost_count += 1

                    log_line("\n[Challenge Follow-up - Percentage Mode] Two-step survivors")
                    log_line("  - Post-pass loss condition: 10% drawdown OR 5% daily drawdown.")
                    if ts_lost_count:
                        log_line(f"  - Avg max profit before loss (%): {ts_sum_max_profit_pct / ts_lost_count:.2f}")
                        log_line(f"  - Avg trades before loss: {ts_sum_trades_to_loss / ts_lost_count:.2f}")
                    else:
                        log_line("  - Avg max profit before loss (%): n/a")
                        log_line("  - Avg trades before loss: n/a")
                    log_line(f"  - Did not hit post-pass loss condition: {ts_no_loss_count}")
                else:
                    live_profits = [p.get("Live_ROI_R", np.nan) for p in portfolio5_rows if np.isfinite(p.get("Live_ROI_R", np.nan))]
                    live_dds = [p.get("Live_MaxDD_R", np.nan) for p in portfolio5_rows if np.isfinite(p.get("Live_MaxDD_R", np.nan))]

                    prof_mask = [v > 0 for v in live_profits]
                    profitable_count = sum(prof_mask)
                    unprofitable_count = len(live_profits) - profitable_count

                    avg_prof_profit = float(np.mean([v for v in live_profits if v > 0])) if profitable_count else np.nan
                    avg_prof_dd = float(np.mean([d for d, ok in zip(live_dds, prof_mask) if ok])) if profitable_count else np.nan
                    avg_unprof_profit = float(np.mean([v for v in live_profits if v <= 0])) if unprofitable_count else np.nan
                    avg_unprof_dd = float(np.mean([d for d, ok in zip(live_dds, prof_mask) if not ok])) if unprofitable_count else np.nan

                    log_line(f"Profitable (live): {profitable_count}")
                    log_line(f"Unprofitable (live): {unprofitable_count}")
                    if profitable_count:
                        log_line(f"Avg profit (profitable, R): {avg_prof_profit:.2f}")
                        log_line(f"Avg max DD (profitable, R): {avg_prof_dd:.2f}")
                    if unprofitable_count:
                        log_line(f"Avg profit (unprofitable, R): {avg_unprof_profit:.2f}")
                        log_line(f"Avg max DD (unprofitable, R): {avg_unprof_dd:.2f}")

                if not USE_PCT_MODE:
                    # Time-to-target test in fixed-risk mode (R units).
                    time_to_5r = []
                    days_to_5r = []
                    candles_to_5r = []
                    hit_5_before_3 = 0
                    hit_5_before_3_and_4day = 0
                    time_to_5r_with_all_filters = []
                    days_to_5r_with_all = []
                    candles_to_5r_with_all = []

                    for pcurve in portfolio_live_curves:
                        live_r_raw = pcurve.get("Live_R", np.array([], dtype=float))
                        live_r = live_r_raw * R_UNIT_SCALE
                        live_eq = np.cumsum(live_r) if live_r.size else np.array([], dtype=float)
                        if live_eq.size == 0:
                            continue
                        t_target = _first_index_ge(live_eq, TARGET_R)
                        t_dd = _first_index_le(live_eq, DRAWDOWN_LIMIT_R)
                        day_breach = _day_loss_breach(pcurve["Live_Sort_Keys"], live_r, limit=DAY_LOSS_LIMIT_R)
                        if t_target is not None:
                            time_to_5r.append(t_target + 1)  # trades are 0-indexed
                            c5, d5, cand5 = _trade_day_candle_to_target(live_eq, pcurve["Live_Sort_Keys"], TARGET_R)
                            if c5 is not None:
                                candles_to_5r.append(cand5)
                            if d5 is not None:
                                days_to_5r.append(d5)
                            if t_dd is None or t_target < t_dd:
                                hit_5_before_3 += 1
                                if day_breach is None or t_target < day_breach:
                                    hit_5_before_3_and_4day += 1
                                    time_to_5r_with_all_filters.append(t_target + 1)
                                    if d5 is not None:
                                        days_to_5r_with_all.append(d5)
                                    if cand5 is not None:
                                        candles_to_5r_with_all.append(cand5)

                    if time_to_5r:
                        log_line(f"\n[Target Test - Fixed R] Portfolios reaching {TARGET_R:.2f}R: {len(time_to_5r)}")
                        log_line(f"  - Avg trades to reach target: {float(np.mean(time_to_5r)):.2f}")
                        if days_to_5r:
                            log_line(f"  - Avg days to reach target (where timestamps exist): {float(np.mean(days_to_5r)):.2f}")
                        if candles_to_5r:
                            log_line(f"  - Avg candles to reach target (trade-count proxy): {float(np.mean(candles_to_5r)):.2f}")
                        log_line(f"  - Reached target before drawdown limit ({DRAWDOWN_LIMIT_R:.2f}R): {hit_5_before_3}")
                        log_line(f"  - Reached target before drawdown and day-loss limit ({DAY_LOSS_LIMIT_R:.2f}R/day): {hit_5_before_3_and_4day}")
                        if time_to_5r_with_all_filters:
                            log_line(f"  - Avg trades to target with all filters: {float(np.mean(time_to_5r_with_all_filters)):.2f}")
                        if days_to_5r_with_all:
                            log_line(f"  - Avg days to target with all filters: {float(np.mean(days_to_5r_with_all)):.2f}")
                        if candles_to_5r_with_all:
                            log_line(f"  - Avg candles to target with all filters: {float(np.mean(candles_to_5r_with_all)):.2f}")
                    else:
                        log_line(f"\n[Target Test - Fixed R] No portfolio reached {TARGET_R:.2f}R in live trades.")

                    # -------------------------------
                    # Challenge-style pass rates (fixed-risk / R)
                    # -------------------------------
                    total_portfolios = len(portfolio_live_curves)
                    ss_count = 0
                    ts_step1_count = 0
                    ts_step2_count = 0

                    def _first_idx_ge_from_r(eq: np.ndarray, thr: float, start_idx: int):
                        if eq is None or eq.size == 0 or start_idx >= eq.size:
                            return None
                        hits = np.where(eq[start_idx:] >= thr)[0]
                        return int(hits[0] + start_idx) if hits.size else None

                    def _first_idx_le_from_r(eq: np.ndarray, thr: float, start_idx: int):
                        if eq is None or eq.size == 0 or start_idx >= eq.size:
                            return None
                        hits = np.where(eq[start_idx:] <= thr)[0]
                        return int(hits[0] + start_idx) if hits.size else None

                    def _day_loss_breach_from_r(sort_keys, r_vals, limit: float, start_idx: int):
                        if not sort_keys or len(sort_keys) != len(r_vals):
                            return None
                        ts = pd.to_datetime(sort_keys, errors="coerce")
                        if ts.isna().all():
                            return None
                        day_sum = 0.0
                        current_day = None
                        for idx, row in pd.DataFrame({"ts": ts, "r": r_vals}).iterrows():
                            if idx < start_idx:
                                continue
                            t = row["ts"]
                            if pd.isna(t):
                                continue
                            d = t.date()
                            if current_day is None or d != current_day:
                                current_day = d
                                day_sum = 0.0
                            day_sum += row["r"]
                            if day_sum <= limit:
                                return int(idx)
                        return None

                    for pcurve in portfolio_live_curves:
                        live_r_raw = pcurve.get("Live_R", np.array([], dtype=float))
                        live_r = live_r_raw * R_UNIT_SCALE
                        live_eq = np.cumsum(live_r) if live_r.size else np.array([], dtype=float)
                        keys = pcurve.get("Live_Sort_Keys", [])
                        if live_eq.size == 0:
                            continue

                        # Single Step: +1.0R before -0.6R and before -0.4R day
                        up_i = _first_index_ge(live_eq, 1.0)
                        dn_i = _first_index_le(live_eq, -0.6)
                        day_i = _day_loss_breach(sort_keys=keys, r_vals=live_r, limit=-0.4)
                        if up_i is not None and (dn_i is None or up_i < dn_i) and (day_i is None or up_i < day_i):
                            ss_count += 1

                        # Two Step: Step 1 = +1.0R before -1.0R and before -0.5R day
                        up1_i = _first_index_ge(live_eq, 1.0)
                        dn1_i = _first_index_le(live_eq, -1.0)
                        day1_i = _day_loss_breach(sort_keys=keys, r_vals=live_r, limit=-0.5)
                        if up1_i is not None and (dn1_i is None or up1_i < dn1_i) and (day1_i is None or up1_i < day1_i):
                            ts_step1_count += 1

                            # Step 2 (after Step 1): +0.5R before -1.0R and before -0.5R day, relative to eq at step1
                            base = live_eq[up1_i]
                            eq_rel = live_eq - base
                            up2_i = _first_idx_ge_from_r(eq_rel, 0.5, up1_i + 1)
                            dn2_i = _first_idx_le_from_r(eq_rel, -1.0, up1_i + 1)
                            day2_i = _day_loss_breach_from_r(keys, live_r, limit=-0.5, start_idx=up1_i + 1)
                            if up2_i is not None and (dn2_i is None or up2_i < dn2_i) and (day2_i is None or up2_i < day2_i):
                                ts_step2_count += 1

                    log_line("\n[Challenge Test - Fixed R Mode] Single-step rule")
                    log_line("  - Requirement: hit +1.0R before -0.6R drawdown and before any -0.4R day.")
                    log_line(f"  - Passes: {ss_count} / {total_portfolios}")
                    log_line("\n[Challenge Test - Fixed R Mode] Two-step rule")
                    log_line("  - Step 1: hit +1.0R before -1.0R drawdown and before any -0.5R day.")
                    log_line("  - Step 2: after Step 1, hit +0.5R before -1.0R drawdown and before any -0.5R day.")
                    log_line(f"  - Step 1 passes: {ts_step1_count} / {total_portfolios}")
                    log_line(f"  - Full two-step passes: {ts_step2_count} / {total_portfolios}")

                    # -------------------------------
                    # Single-step follow-up stats (fixed-risk / R)
                    # -------------------------------
                    ss_lost_count = 0
                    ss_no_loss_count = 0
                    sum_max_profit_r = 0.0
                    sum_trades_to_loss = 0.0

                    for pcurve in portfolio_live_curves:
                        live_r_raw = pcurve.get("Live_R", np.array([], dtype=float))
                        live_r = live_r_raw * R_UNIT_SCALE
                        live_eq = np.cumsum(live_r) if live_r.size else np.array([], dtype=float)
                        keys = pcurve.get("Live_Sort_Keys", [])
                        if live_eq.size == 0:
                            continue

                        up_i = _first_index_ge(live_eq, 1.0)
                        dn_i = _first_index_le(live_eq, -0.6)
                        day_i = _day_loss_breach(sort_keys=keys, r_vals=live_r, limit=-0.4)
                        if up_i is None or (dn_i is not None and dn_i < up_i) or (day_i is not None and day_i < up_i):
                            continue

                        loss_dd_i = _first_idx_le_from_r(live_eq, -1.0, up_i + 1)
                        loss_day_i = _day_loss_breach_from_r(keys, live_r, limit=-0.5, start_idx=up_i + 1)
                        loss_candidates = [i for i in [loss_dd_i, loss_day_i] if i is not None]
                        if not loss_candidates:
                            ss_no_loss_count += 1
                            continue

                        loss_i = min(loss_candidates)
                        max_eq = float(np.max(live_eq[:loss_i + 1])) if loss_i is not None else float(np.max(live_eq))
                        sum_max_profit_r += max_eq
                        sum_trades_to_loss += (loss_i + 1)
                        ss_lost_count += 1

                    log_line("\n[Challenge Follow-up - Fixed R Mode] Single-step survivors")
                    log_line("  - Post-pass loss condition: -1.0R drawdown OR -0.5R daily drawdown.")
                    if ss_lost_count:
                        log_line(f"  - Avg max profit before loss (R): {sum_max_profit_r / ss_lost_count:.2f}")
                        log_line(f"  - Avg trades before loss: {sum_trades_to_loss / ss_lost_count:.2f}")
                    else:
                        log_line("  - Avg max profit before loss (R): n/a (no portfolio hit the post-pass loss condition)")
                        log_line("  - Avg trades before loss: n/a (no portfolio hit the post-pass loss condition)")
                    log_line(f"  - Did not hit post-pass loss condition: {ss_no_loss_count}")

                    # -------------------------------
                    # Two-step follow-up stats (fixed-risk / R)
                    # -------------------------------
                    ts_lost_count = 0
                    ts_no_loss_count = 0
                    ts_sum_max_profit_r = 0.0
                    ts_sum_trades_to_loss = 0.0

                    for pcurve in portfolio_live_curves:
                        live_r_raw = pcurve.get("Live_R", np.array([], dtype=float))
                        live_r = live_r_raw * R_UNIT_SCALE
                        live_eq = np.cumsum(live_r) if live_r.size else np.array([], dtype=float)
                        keys = pcurve.get("Live_Sort_Keys", [])
                        if live_eq.size == 0:
                            continue

                        # Step 1
                        up1_i = _first_index_ge(live_eq, 1.0)
                        dn1_i = _first_index_le(live_eq, -1.0)
                        day1_i = _day_loss_breach(sort_keys=keys, r_vals=live_r, limit=-0.5)
                        if up1_i is None or (dn1_i is not None and dn1_i < up1_i) or (day1_i is not None and day1_i < up1_i):
                            continue

                        # Step 2 (after step 1), relative to eq at step1
                        base = live_eq[up1_i]
                        eq_rel = live_eq - base
                        up2_i = _first_idx_ge_from_r(eq_rel, 0.5, up1_i + 1)
                        dn2_i = _first_idx_le_from_r(eq_rel, -1.0, up1_i + 1)
                        day2_i = _day_loss_breach_from_r(keys, live_r, limit=-0.5, start_idx=up1_i + 1)
                        if up2_i is None or (dn2_i is not None and dn2_i < up2_i) or (day2_i is not None and day2_i < up2_i):
                            continue

                        # Loss after completing step 2 (absolute loss rules)
                        loss_dd_i = _first_idx_le_from_r(live_eq, -1.0, up2_i + 1)
                        loss_day_i = _day_loss_breach_from_r(keys, live_r, limit=-0.5, start_idx=up2_i + 1)
                        loss_candidates = [i for i in [loss_dd_i, loss_day_i] if i is not None]
                        if not loss_candidates:
                            ts_no_loss_count += 1
                            continue

                        loss_i = min(loss_candidates)
                        max_eq = float(np.max(live_eq[:loss_i + 1]))
                        ts_sum_max_profit_r += max_eq
                        ts_sum_trades_to_loss += (loss_i + 1)
                        ts_lost_count += 1

                    log_line("\n[Challenge Follow-up - Fixed R Mode] Two-step survivors")
                    log_line("  - Post-pass loss condition: -1.0R drawdown OR -0.5R daily drawdown.")
                    if ts_lost_count:
                        log_line(f"  - Avg max profit before loss (R): {ts_sum_max_profit_r / ts_lost_count:.2f}")
                        log_line(f"  - Avg trades before loss: {ts_sum_trades_to_loss / ts_lost_count:.2f}")
                    else:
                        log_line("  - Avg max profit before loss (R): n/a (no portfolio hit the post-pass loss condition)")
                        log_line("  - Avg trades before loss: n/a (no portfolio hit the post-pass loss condition)")
                    log_line(f"  - Did not hit post-pass loss condition: {ts_no_loss_count}")

                # -------------------------------
                # Plot: median live equity curve + best/worst bands (Top 100 by Hist PF)
                # -------------------------------
                if ENABLE_PLOTS and portfolio_live_curves:
                    top_curves = sorted(
                        [p for p in portfolio_live_curves if np.isfinite(p.get("Hist_PF", np.nan))],
                        key=lambda x: x["Hist_PF"],
                        reverse=True
                    )[:100]
                    if top_curves:
                        live_series = []
                        finals = []
                        max_len = 0
                        for p in top_curves:
                            live_r_raw = p.get("Live_R", np.array([], dtype=float))
                            if USE_PCT_MODE:
                                eqm = np.cumprod(1.0 + (live_r_raw * RISK_PCT_PER_R)) if live_r_raw.size else np.array([], dtype=float)
                                series = (eqm - 1.0) * 100.0
                            else:
                                series = np.cumsum(live_r_raw * R_UNIT_SCALE) if live_r_raw.size else np.array([], dtype=float)
                            live_series.append(series)
                            max_len = max(max_len, len(series))
                            finals.append(series[-1] if series.size else np.nan)

                        if max_len > 0:
                            padded = np.full((len(top_curves), max_len), np.nan, dtype=float)
                            for i, series in enumerate(live_series):
                                if series is None or len(series) == 0:
                                    continue
                                padded[i, :len(series)] = series
                            x = np.arange(max_len)

                            finals_valid = [(v, i) for i, v in enumerate(finals) if np.isfinite(v)]
                            best_idx = max(finals_valid, key=lambda t: t[0])[1] if finals_valid else None
                            worst_idx = min(finals_valid, key=lambda t: t[0])[1] if finals_valid else None
                            if finals_valid:
                                finals_sorted = sorted(finals_valid, key=lambda t: t[0])
                                median_idx = finals_sorted[len(finals_sorted) // 2][1]
                            else:
                                median_idx = None

                            plt.figure()
                            if best_idx is not None:
                                plt.plot(x, padded[best_idx], color="green", linewidth=1.5, label="Best live (top100 PF)")
                            if worst_idx is not None:
                                plt.plot(x, padded[worst_idx], color="red", linewidth=1.5, label="Worst live (top100 PF)")
                            if median_idx is not None:
                                plt.plot(x, padded[median_idx], color="navy", linewidth=2, label="Median live (top100 PF)")
                            plt.title("Top 100 Portfolios (Hist PF) - Live Equity Curves")
                            plt.xlabel("Live trade index")
                            plt.ylabel("Live Return (%)" if USE_PCT_MODE else "Cumulative R (live only)")
                            plt.legend()
                            plt.tight_layout()
                            plt.show()
                elif ENABLE_PLOTS:
                    log_line("\n[Equity plot skipped] No usable live equity curves were built.\n")

# -------------------------------
