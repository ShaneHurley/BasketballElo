#!/usr/bin/env python3
"""Apply leak-fix patches to the original Jupyter notebook."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "Copy_of_nba_unified_prediction_pipeline (35).ipynb"


def patch_source(src: str) -> str:
    replacements = [
        ("margin_A = end_A - start_B", "margin_A = end_A - start_A"),
        (
            "            delta = error * 0.9 * (shares[i] / s) * k_mult\n",
            "            k_base = self.cfg.get(\"K_OFF\", 0.9) if side == \"off\" else self.cfg.get(\"K_DEF\", 0.9)\n"
            "            rd = pl[\"O_rd\"] if side == \"off\" else pl[\"D_rd\"]\n"
            "            k_effective = k_base * (rd / 350.0)\n"
            "            delta = error * k_effective * (shares[i] / s) * k_mult\n",
        ),
        (
            '"ELO_SCALING_FACTOR": 1000,\n    }',
            '"ELO_SCALING_FACTOR": 1000,\n        "K_OFF": 0.9,\n        "K_DEF": 0.9,\n    }',
        ),
        (
            "pred_ppp_H = league_xppp + home_boost + (ho_off - ao_off) / elo_scaling",
            "pred_ppp_H = league_xppp + home_boost + (ho_off - ao_def) / elo_scaling",
        ),
        (
            "    return study.best_params\n\n\ndef tune_hierarchical",
            "    from pipeline.utils import map_elo_params\n    return map_elo_params(study.best_params)\n\n\ndef tune_hierarchical",
        ),
        (
            "        best_elo = tune_elo_tracker(train_stints, DEFAULT_LEAGUE_XPPP, n_trials=n_tuning_trials_elo)",
            "        best_elo = tune_elo_tracker(train_stints, DEFAULT_LEAGUE_XPPP, n_trials=n_tuning_trials_elo)  # mapped config",
        ),
        (
            "        sim_pace = PaceTracker(team_window=10, league_window=150)   # fresh for test",
            "        sim_pace = copy.deepcopy(full_pace)\n        sim_xppp = copy.deepcopy(full_xppp)",
        ),
        (
            "            calibrator=rolling_calibrator\n        )",
            "            calibrator=rolling_calibrator,\n"
            "            team_xppp_tracker=sim_xppp,\n"
            "            spread_calibrator=spread_calibrator,\n        )",
        ),
        (
            "            preds = meta_model.predict(feat)",
            "            from pipeline.model import build_feature_row\n            preds = meta_model.predict(build_feature_row(feat))",
        ),
        ("sos_window: int = 10,", "sos_window: int = 15,"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)
    return src


def main():
    nb = json.loads(NOTEBOOK.read_text())
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        original = "".join(cell.get("source", []))
        patched = patch_source(original)
        if patched != original:
            cell["source"] = [line + "\n" for line in patched.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            changed += 1

    # Add pipeline import cell after first markdown if missing
    import_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Leak-fixed pipeline (optional: use instead of inline cells)\n",
            "import sys\n",
            "sys.path.insert(0, '.')\n",
            "from pipeline.model import build_feature_row\n",
            "from pipeline.backtest import run_multi_year_backtest_walkforward, grid_search_bet_edge, benchmark_results\n",
            "from pipeline.predict import predict_game\n",
            "from pipeline.ratings import PlayerRatingTracker\n",
            "from pipeline.config import STATE_DIR, SOS_WINDOW, OPTIMAL_BET_EDGE\n",
        ],
        "outputs": [],
        "execution_count": None,
    }
    nb["cells"].insert(1, import_cell)
    changed += 1

    NOTEBOOK.write_text(json.dumps(nb, indent=2))
    print(f"Patched notebook: {changed} cells updated -> {NOTEBOOK}")


if __name__ == "__main__":
    main()
