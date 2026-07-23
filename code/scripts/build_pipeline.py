#!/usr/bin/env python3
"""Assemble pipeline package from notebook extracts with leak fixes applied."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACT = ROOT / "pipeline" / "_extract"
OUT = ROOT / "pipeline"


def read_cell(n: int) -> str:
    return (EXTRACT / f"cell_{n:02d}.py").read_text()


def strip_prints(src: str) -> str:
  lines = []
  for line in src.splitlines():
    if line.strip().startswith("print(") and "✅" in line:
      continue
    lines.append(line)
  return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # config.py
    cell04 = read_cell(4)
    config_start = cell04.index("# ─── Constants")
    config_end = cell04.index("# ─── Team name")
    constants = cell04[config_start:config_end]
    constants += "\nSOS_WINDOW = 15\nOPTIMAL_BET_EDGE = 2.5  # tuned via grid search\nSPREAD_CALIB_WINDOW = 50\n"
    (OUT / "config.py").write_text(
        '"""Pipeline configuration and constants."""\nfrom pathlib import Path\n\n'
        + cell04[:cell04.index("# ─── Google Colab")]
        + constants
        + "\n"
    )

    # utils.py - player parsing from cell 6
    cell06 = read_cell(6)
    utils_end = cell06.index("# ─────────────────────────────────────────────────────────────────────\n# 2025-26")
    (OUT / "utils.py").write_text(
        '"""Shared helpers."""\nimport math\nimport numpy as np\nimport pandas as pd\n\n'
        + cell06[:utils_end]
        + """

def map_elo_params(optuna_params: dict) -> dict:
    \"\"\"Map Optuna Elo tuner keys to PlayerRatingTracker config keys.\"\"\"
    return {
        "K_OFF": optuna_params["k_off"],
        "K_DEF": optuna_params["k_def"],
        "ELO_SCALING_FACTOR": optuna_params["elo_scaling"],
        "HOME_PPP_BOOST": optuna_params["home_boost"],
        "OFFSEASON_REVERSION": optuna_params["offseason_reversion"],
        "USAGE_FLOOR": optuna_params["usage_floor"],
        "assist_split": optuna_params["assist_split"],
    }
"""
    )

    # ingest.py - converters
    ingest = cell06[utils_end:]
    (OUT / "ingest.py").write_text(
        '"""PBP format converters."""\nfrom pipeline.utils import _map_player_name\n\n' + ingest
    )

    # preprocess.py - cell 8 with FT fix
    preprocess = read_cell(8)
    preprocess = preprocess.replace(
        "    ft_att = (df[\"EVENTMSGTYPE\"] == 3)\n"
        "    ft_pct = ft_made.sum() / ft_att.sum() if ft_att.sum() > 0 else 0.77\n"
        "    df.loc[ft_att & is_home, \"home_xpts_added\"] = ft_pct\n"
        "    df.loc[ft_att & is_away, \"away_xpts_added\"] = ft_pct\n",
        """    ft_att = (df["EVENTMSGTYPE"] == 3)
    # Walk-forward FT%: each game uses only prior games' free throws (no season leak)
    if "GAME_ID" in df.columns and "game_date" in df.columns:
        sort_cols = ["game_date", "GAME_ID"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        game_order = df.groupby("GAME_ID", sort=False).ngroup()
        ft_made_flag = ft_made.astype(int)
        ft_att_flag = ft_att.astype(int)
        cum_made = ft_made_flag.groupby(game_order).cumsum().shift(1, fill_value=0)
        cum_att = ft_att_flag.groupby(game_order).cumsum().shift(1, fill_value=0)
        game_cum = df.groupby("GAME_ID", sort=False).apply(
            lambda g: pd.Series({
                "gm_made": ft_made.loc[g.index].sum(),
                "gm_att": ft_att.loc[g.index].sum(),
            })
        )
        game_cum.index = game_cum.index.get_level_values(0) if hasattr(game_cum.index, 'levels') else game_cum.index
        prior_made = game_order.map(
            lambda g: game_cum.loc[g, "gm_made"] if g in game_cum.index else 0
        ) if False else None
        # Expanding prior-game FT% per row
        running_made = 0
        running_att = 0
        ft_pct_by_row = np.full(len(df), 0.77)
        seen_games = set()
        last_gid = None
        for idx, row in df.iterrows():
            gid = row["GAME_ID"]
            if gid != last_gid and last_gid is not None:
                gm = df[df["GAME_ID"] == last_gid]
                running_made += ft_made.loc[gm.index].sum()
                running_att += ft_att.loc[gm.index].sum()
            if running_att > 0:
                ft_pct_by_row[idx] = running_made / running_att
            last_gid = gid
        df["_ft_pct_prior"] = ft_pct_by_row
        df.loc[ft_att & is_home, "home_xpts_added"] = df.loc[ft_att & is_home, "_ft_pct_prior"]
        df.loc[ft_att & is_away, "away_xpts_added"] = df.loc[ft_att & is_away, "_ft_pct_prior"]
        df.drop(columns=["_ft_pct_prior"], inplace=True)
    else:
        ft_pct = 0.77
        df.loc[ft_att & is_home, "home_xpts_added"] = ft_pct
        df.loc[ft_att & is_away, "away_xpts_added"] = ft_pct
""",
    )
    (OUT / "preprocess.py").write_text(
        '"""PBP preprocessing and xPoints."""\nimport re\nimport numpy as np\nimport pandas as pd\n\n'
        + preprocess
    )

    # stints.py
    (OUT / "stints.py").write_text(
        '"""Lineup stint builder."""\nfrom pipeline.config import ASSIST_SPLIT\n\n' + read_cell(10)
    )

    # ratings.py - fixed PlayerRatingTracker
    ratings = read_cell(13)
    ratings = ratings.replace(
        '"ELO_SCALING_FACTOR": 1000,\n    }',
        '"ELO_SCALING_FACTOR": 1000,\n        "K_OFF": 0.9,\n        "K_DEF": 0.9,\n    }',
    )
    ratings = ratings.replace("margin_A = end_A - start_B", "margin_A = end_A - start_A")
    ratings = ratings.replace(
        "            delta = error * 0.9 * (shares[i] / s) * k_mult\n",
        """            k_base = self.cfg.get("K_OFF", 0.9) if side == "off" else self.cfg.get("K_DEF", 0.9)
            rd = pl["O_rd"] if side == "off" else pl["D_rd"]
            k_effective = k_base * (rd / 350.0)
            delta = error * k_effective * (shares[i] / s) * k_mult
""",
    )
    ratings = ratings.replace(
        """    @staticmethod
    def _weight_stint(poss, period, margin, season_progress):
        \"\"\"Weight a stint by possessions, downweighting garbage time.\"\"\"
        weight = poss
        if period >= 4 and abs(margin) >= 15:
            weight *= 0.5
        return weight""",
        """    @staticmethod
    def _weight_stint(poss, period, margin, season_progress):
        \"\"\"Weight a stint by possessions, downweighting garbage time.\"\"\"
        weight = poss
        if period >= 4 and abs(margin) >= 15:
            weight *= 0.5
        if abs(margin) >= 25:
            weight *= 0.5
        weight *= (0.7 + 0.3 * season_progress)
        return weight

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"players": self.players, "player_games": dict(self.player_games), "cfg": self.cfg}, f)

    @classmethod
    def load_state(cls, path, league_xppp=1.10):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        tracker = cls(config=data.get("cfg"), league_xppp=league_xppp)
        tracker.players = data["players"]
        tracker.player_games = defaultdict(int, data.get("player_games", {}))
        return tracker""",
    )
    ratings = strip_prints(ratings)
    (OUT / "ratings.py").write_text(
        '"""Player-level Elo/xPPP rating tracker."""\n' + ratings
    )

    # hierarchical.py
    hier = strip_prints(read_cell(15))
    hier += """

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj
"""
    (OUT / "hierarchical.py").write_text('"""Hierarchical possession engine."""\n' + hier)

    # market.py - cell 17 + SpreadCalibrator parts
    market = read_cell(17)
    (OUT / "market.py").write_text(
        '"""Odds loading and betting helpers."""\nfrom pipeline.config import GOOD_BET_EDGE, TEAM_MAP\n\n'
        + market
    )

    # trackers.py - cell 18 + cell 28 PaceTracker
    (OUT / "trackers.py").write_text(
        '"""Rolling team trackers."""\nfrom pipeline.config import DEFAULT_LEAGUE_XPPP\n\n'
        + read_cell(18)
        + "\n\n"
        + read_cell(28)
    )

    # features.py - cell 21
    features = read_cell(21)
    features = features.replace("sos_window: int = 15", "sos_window: int = None")
    features = features.replace(
        "    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))",
        "    from pipeline.config import SOS_WINDOW\n    if sos_window is None:\n        sos_window = SOS_WINDOW\n    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))",
    )
    (OUT / "features.py").write_text(
        '"""Feature generation for training and inference."""\n'
        + features
    )

    # model.py - engineer_interaction + MetaScoreModel from cell 23
    model_src = read_cell(23)
    model_src += """

def build_feature_row(raw_feat: dict) -> dict:
    \"\"\"Apply interaction features to a single pre-game feature dict (train/serve parity).\"\"\"
    df = pd.DataFrame([raw_feat])
    return engineer_interaction_features(df).iloc[0].to_dict()
"""
    (OUT / "model.py").write_text('"""Meta-model and feature engineering."""\n' + model_src)

    # tuning.py - cell 26 with fixes
    tuning = read_cell(26)
    tuning = tuning.replace(
        "pred_ppp_H = league_xppp + home_boost + (ho_off - ao_off) / elo_scaling",
        "pred_ppp_H = league_xppp + home_boost + (ho_off - ao_def) / elo_scaling",
    )
    tuning = tuning.replace(
        "    return study.best_params\n",
        "    return map_elo_params(study.best_params)\n",
        1,
    )
    (OUT / "tuning.py").write_text(
        '"""Hyperparameter tuning."""\nfrom pipeline.utils import map_elo_params\n\n' + tuning
    )

    # calibrators.py
    (OUT / "calibrators.py").write_text(read_cell(24) + "\n" + read_cell(29))

    # simulate.py - cell 30 with fixes
    sim = read_cell(30)
    sim = sim.replace("sos_window: int = 10,", "sos_window: int = None,")
    sim = sim.replace(
        "    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))",
        "    from pipeline.config import SOS_WINDOW\n    if sos_window is None:\n        sos_window = SOS_WINDOW\n    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))",
    )
    sim = sim.replace(
        "        # Prediction\n        if hasattr(meta_model, 'fitted') and meta_model.fitted:\n"
        "            preds = meta_model.predict(feat)",
        "        # Prediction (with interaction features for train/serve parity)\n"
        "        feat_model = build_feature_row(feat)\n"
        "        if hasattr(meta_model, 'fitted') and meta_model.fitted:\n"
        "            preds = meta_model.predict(feat_model)",
    )
    sim = sim.replace(
        "from tqdm.auto import tqdm",
        "from tqdm.auto import tqdm\nfrom pipeline.model import build_feature_row\nfrom pipeline.config import GOOD_BET_EDGE, DEFAULT_LEAGUE_XPPP, ALTITUDE_TEAMS, SOS_WINDOW",
    )
    sim = sim.replace(
        "            \"KELLY_FRACTION\": round(kelly_fraction, 4),\n        })",
        """            "KELLY_FRACTION": round(kelly_fraction, 4),
            "MODEL_ML_CORRECT": int((preds.get("win_prob", 0.5) > 0.5) == (act_h - act_a > 0)),
            "TOTAL_ERR": round(preds.get("pred_total", 0) - (act_h + act_a), 2),
        })""",
    )
    (OUT / "simulate.py").write_text(sim)

    # backtest.py - cell 38 with all wiring fixes
    bt = read_cell(38)
    bt = bt.replace(
        "        best_elo = tune_elo_tracker(train_stints, DEFAULT_LEAGUE_XPPP, n_trials=n_tuning_trials_elo)",
        "        best_elo = tune_elo_tracker(train_stints, DEFAULT_LEAGUE_XPPP, n_trials=n_tuning_trials_elo)  # returns mapped config",
    )
    bt = bt.replace(
        "        base_pace = PaceTracker(team_window=10, league_window=100)\n\n        base_features = generate_features(\n"
        "            base_stints, base_hier, base_elo, base_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True\n        )",
        "        base_pace = PaceTracker(team_window=10, league_window=100)\n"
        "        base_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)\n\n"
        "        base_features = generate_features(\n"
        "            base_stints, base_hier, base_elo, base_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True,\n"
        "            team_xppp_tracker=base_xppp,\n"
        "        )",
    )
    bt = bt.replace(
        "        calib_pace = copy.deepcopy(base_pace)\n\n        calib_features = generate_features(\n"
        "            calib_stints, calib_hier, calib_elo, calib_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True     # Engines evolve realistically\n        )",
        "        calib_pace = copy.deepcopy(base_pace)\n"
        "        calib_xppp = copy.deepcopy(base_xppp)\n\n"
        "        calib_features = generate_features(\n"
        "            calib_stints, calib_hier, calib_elo, calib_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True,\n"
        "            team_xppp_tracker=calib_xppp,\n"
        "        )",
    )
    bt = bt.replace(
        "        full_pace = PaceTracker(team_window=10, league_window=100)\n"
        "        _ = generate_features(\n"
        "            train_stints, full_hier, full_elo, full_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True\n        )",
        "        full_pace = PaceTracker(team_window=10, league_window=100)\n"
        "        full_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)\n"
        "        _ = generate_features(\n"
        "            train_stints, full_hier, full_elo, full_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            update_engines=True,\n"
        "            team_xppp_tracker=full_xppp,\n"
        "        )",
    )
    bt = bt.replace(
        "        sim_pace = PaceTracker(team_window=10, league_window=150)   # fresh for test",
        "        sim_pace = copy.deepcopy(full_pace)\n"
        "        sim_xppp = copy.deepcopy(full_xppp)",
    )
    bt = bt.replace(
        "        spread_calibrator = SpreadCalibrator(window=50)",
        "        from pipeline.config import SPREAD_CALIB_WINDOW\n"
        "        spread_calibrator = SpreadCalibrator(window=SPREAD_CALIB_WINDOW)",
    )
    bt = bt.replace(
        "        results = run_simulation(\n"
        "            season_df=test_stints,\n"
        "            hier_engine=full_hier,\n"
        "            elo_tracker=full_elo,\n"
        "            meta_model=meta_model,\n"
        "            pace_tracker=sim_pace,\n"
        "            odds_dict=odds_dict,                # safe get_odds inside\n"
        "            calibrator=rolling_calibrator\n        )",
        "        from pipeline.config import GOOD_BET_EDGE\n"
        "        results = run_simulation(\n"
        "            season_df=test_stints,\n"
        "            hier_engine=full_hier,\n"
        "            elo_tracker=full_elo,\n"
        "            meta_model=meta_model,\n"
        "            pace_tracker=sim_pace,\n"
        "            odds_dict=odds_dict,\n"
        "            calibrator=rolling_calibrator,\n"
        "            team_xppp_tracker=sim_xppp,\n"
        "            spread_calibrator=spread_calibrator,\n"
        "        )",
    )
    bt += """

def grid_search_bet_edge(results_df, edges=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0)):
    \"\"\"Find ATS edge threshold that maximizes ROI at -110 on walk-forward results.\"\"\"
    best_edge, best_roi = 2.5, -1.0
    rows = []
    for edge in edges:
        mask = results_df["EDGE"].abs() >= edge
        bets = results_df[mask & (results_df["DIRECTION"] != "Pass")]
        if len(bets) < 20:
            continue
        wins = ((bets["DIRECTION"] == "Home") & (bets["ACTUAL_MARGIN"] + bets["MARKET_SPREAD"] > 0)) | \\
               ((bets["DIRECTION"] == "Away") & (bets["ACTUAL_MARGIN"] + bets["MARKET_SPREAD"] < 0))
        win_pct = wins.mean()
        roi = win_pct * (100 / 110) - (1 - win_pct)
        rows.append({"edge": edge, "n_bets": len(bets), "win_pct": win_pct, "roi": roi})
        if roi > best_roi:
            best_roi, best_edge = roi, edge
    return best_edge, pd.DataFrame(rows)


def benchmark_results(results_df):
    \"\"\"Print spread MAE, ATS, ML diagnostic, total MAE.\"\"\"
    if results_df.empty:
        print("No results to benchmark.")
        return
    mae = (results_df["PRED_SPREAD"] - results_df["ACTUAL_MARGIN"]).abs().mean()
    active = results_df[results_df["DIRECTION"] != "Pass"]
    if len(active):
        ats = (
            ((active["DIRECTION"] == "Home") & (active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"] > 0))
            | ((active["DIRECTION"] == "Away") & (active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"] < 0))
        ).mean()
    else:
        ats = float("nan")
    ml_acc = results_df.get("MODEL_ML_CORRECT", pd.Series(dtype=float)).mean()
    total_mae = results_df.get("TOTAL_ERR", pd.Series(dtype=float)).abs().mean()
    print(f"Spread MAE: {mae:.2f}")
    print(f"ATS win% (active bets): {ats:.1%} ({len(active)} bets)")
    print(f"ML winner accuracy (diag): {ml_acc:.1%}")
    print(f"Total MAE (diag): {total_mae:.2f}")
"""
    (OUT / "backtest.py").write_text(
        '"""Walk-forward backtest orchestration."""\n'
        "import copy\nimport pandas as pd\nimport numpy as np\n"
        "from catboost import CatBoostRegressor\n"
        "from sklearn.linear_model import Ridge, HuberRegressor\n"
        "from sklearn.ensemble import StackingRegressor\n"
        "from pipeline.config import DEFAULT_LEAGUE_XPPP\n"
        "from pipeline.ratings import PlayerRatingTracker\n"
        "from pipeline.hierarchical import HierarchicalPossessionEngine\n"
        "from pipeline.trackers import PaceTracker, TeamXpppTracker\n"
        "from pipeline.features import generate_features\n"
        "from pipeline.model import MetaScoreModel, engineer_interaction_features\n"
        "from pipeline.tuning import tune_elo_tracker, tune_hierarchical, tune_meta_model\n"
        "from pipeline.market import SpreadCalibrator\n"
        "from pipeline.calibrators import RollingPlattCalibrator\n"
        "from pipeline.simulate import run_simulation\n\n"
        + bt
    )

    # predict.py
    (OUT / "predict.py").write_text(
        '"""Single-game live prediction."""\n'
        + read_cell(34)  # will patch below via separate write
    )

    # __init__.py
    (OUT / "__init__.py").write_text('"""NBA spread prediction pipeline."""\n')

    print("Pipeline modules written to", OUT)


if __name__ == "__main__":
    main()
