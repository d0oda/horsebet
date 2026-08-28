import pytest
from unittest.mock import patch, MagicMock
from scraper.odds_watcher import (
    _clean_odds_val,
    _format_combination,
    fetch_win_odds,
    fetch_place_odds,
    fetch_exotic_odds,
    fetch_race_odds,
)


def test_clean_odds_val():
    assert _clean_odds_val("3.5") == 3.5
    assert _clean_odds_val("1,250.4") == 1250.4
    assert _clean_odds_val(4.2) == 4.2
    assert _clean_odds_val("---") is None
    assert _clean_odds_val("取消") is None
    assert _clean_odds_val("除外") is None
    assert _clean_odds_val("0") is None
    assert _clean_odds_val("0.0") is None
    assert _clean_odds_val(None) is None


def test_format_combination():
    assert _format_combination("01") == "1"
    assert _format_combination("09") == "9"
    assert _format_combination("0102") == "1-2"
    assert _format_combination("0512") == "5-12"
    assert _format_combination("010203") == "1-2-3"
    assert _format_combination("031418") == "3-14-18"


@patch("scraper.odds_watcher._request_odds_api")
def test_fetch_win_odds(mock_api):
    mock_api.return_value = {
        "odds": {
            "1": {
                "01": ["125.9", "0.0", "15"],
                "02": ["4.5", "0.0", "2"],
                "03": ["---", "0.0", "0"],
            }
        }
    }
    odds = fetch_win_odds("202405021211")
    assert len(odds) == 2
    assert odds[0] == {"combination": "1", "odds_value": 125.9, "popularity": 15, "bet_type": "win"}
    assert odds[1] == {"combination": "2", "odds_value": 4.5, "popularity": 2, "bet_type": "win"}


@patch("scraper.odds_watcher._request_odds_api")
def test_fetch_place_odds(mock_api):
    mock_api.return_value = {
        "odds": {
            "2": {
                "01": ["17.0", "31.4", "16"],
                "02": ["1.5", "2.2", "2"],
                "03": ["取消", "取消", "0"],
            }
        }
    }
    place = fetch_place_odds("202405021211")
    assert len(place) == 2
    assert place[0] == {"combination": "1", "min_odds": 17.0, "max_odds": 31.4, "popularity": 16, "bet_type": "place"}
    assert place[1] == {"combination": "2", "min_odds": 1.5, "max_odds": 2.2, "popularity": 2, "bet_type": "place"}


@patch("scraper.odds_watcher._request_odds_api")
def test_fetch_exotic_odds(mock_api):
    mock_api.return_value = {
        "odds": {
            "4": {
                "0102": ["407.9", "0.0", "55"],
                "0103": ["2,612.8", "0.0", "121"],
            }
        }
    }
    quinella = fetch_exotic_odds("202405021211", bet_types=["quinella"])
    assert len(quinella) == 2
    assert quinella[0] == {"combination": "1-2", "odds_value": 407.9, "popularity": 55, "bet_type": "quinella"}
    assert quinella[1] == {"combination": "1-3", "odds_value": 2612.8, "popularity": 121, "bet_type": "quinella"}


@patch("scraper.odds_watcher._request_odds_api")
def test_fetch_race_odds(mock_api):
    mock_api.return_value = {
        "odds": {
            "1": {"01": ["5.0", "0.0", "1"]},
            "2": {"01": ["1.8", "2.4", "1"]},
        }
    }
    pkg = fetch_race_odds("202405021211", include_exotics=False)
    assert pkg["race_id"] == "202405021211"
    assert len(pkg["win"]) == 1
    assert len(pkg["place"]) == 1
    assert "captured_at" in pkg
