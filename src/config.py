# -*- coding: utf-8 -*-
"""Project paths and external data-source configuration.

`PROJECT_ROOT` is resolved from this file's location, so the code is
location-independent. External data directories come from environment
variables; create an untracked `src/local_config.py` defining the same names
to override them locally.

    IHC_INDEX_1M      per-day parquet, 1-minute index bars
                      (datetime, code, open, high, low, close, volume, money)
    IHC_FUTURES_1M    per-day parquet, 1-minute index-futures bars
    IHC_DOMINANT_MAP  per-day parquet, product -> dominant contract
    IHC_OPTIONS       per-day parquet, index-option ticks with order book
    IHC_KRONOS_REPO   local clone of the Kronos model repository
    IHC_BREADTH_PANEL optional: external cross-sectional breadth panel
                      (date, ti, cs_*) used only by src/orth_check.py
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'out')

_DEFAULTS = dict(INDEX_1M_DIR='/path/to/index_1min',
                 INDEX_DAILY_DIR='/path/to/index_daily',
                 INDEX_INFO='/path/to/index_info.parquet',
                 FUTURES_1M_DIR='/path/to/futures_1min',
                 DOMINANT_MAP_DIR='/path/to/dominant_map',
                 OPTIONS_DIR='/path/to/options_tick',
                 KRONOS_REPO='/path/to/Kronos',
                 BREADTH_PANEL='/path/to/cs_breadth_panel.parquet')
_ENV = dict(INDEX_1M_DIR='IHC_INDEX_1M',
            INDEX_DAILY_DIR='IHC_INDEX_DAILY', INDEX_INFO='IHC_INDEX_INFO', FUTURES_1M_DIR='IHC_FUTURES_1M',
            DOMINANT_MAP_DIR='IHC_DOMINANT_MAP', OPTIONS_DIR='IHC_OPTIONS',
            KRONOS_REPO='IHC_KRONOS_REPO', BREADTH_PANEL='IHC_BREADTH_PANEL')
try:
    from local_config import *          # noqa: F401,F403  (untracked)
except ImportError:
    pass
for _k, _env in _ENV.items():
    if _k not in globals():
        globals()[_k] = os.environ.get(_env, _DEFAULTS[_k])
