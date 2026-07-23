"""NBA spread prediction pipeline (leak-fixed)."""
from pipeline.backtest import run_multi_year_backtest_walkforward
from pipeline.metrics import benchmark_results, grid_search_bet_edge
from pipeline.model import MetaScoreModel, build_feature_row, engineer_interaction_features
from pipeline.predict import predict_game
from pipeline.utils import map_elo_params

__all__ = [
    "run_multi_year_backtest_walkforward",
    "benchmark_results",
    "grid_search_bet_edge",
    "MetaScoreModel",
    "build_feature_row",
    "engineer_interaction_features",
    "predict_game",
    "map_elo_params",
]
