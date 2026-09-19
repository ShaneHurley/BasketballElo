"""Multi-year window / season-split helpers."""
from __future__ import annotations

from pipeline.data_paths import clamp_rolling_window, split_rating_and_model_seasons


def test_clamp_rolling_window_allows_4_and_5():
    assert clamp_rolling_window(4) == 4
    assert clamp_rolling_window(5) == 5
    assert clamp_rolling_window(1) == 2
    assert clamp_rolling_window(9) == 5


def test_split_rating_and_model_seasons():
    seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    rating, model = split_rating_and_model_seasons(seasons, 5, 4, rating_history_use_all=True)
    assert model == [2019, 2020, 2021, 2022]
    assert rating == [2018, 2019, 2020, 2021, 2022]
    rating2, model2 = split_rating_and_model_seasons(seasons, 5, 4, rating_history_use_all=False)
    assert rating2 == model2 == [2019, 2020, 2021, 2022]


def test_split_empty_when_no_prior():
    rating, model = split_rating_and_model_seasons([2020], 0, 4)
    assert rating == [] and model == []
