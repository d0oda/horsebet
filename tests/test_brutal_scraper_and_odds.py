import pytest
from bs4 import BeautifulSoup
from scraper.netkeiba import _extract_race_class, _parse_time, _parse_weight, _parse_odds_text
from scraper.odds_watcher import fetch_win_odds

class TestScraperAndOddsBrutal:
    """Stress tests scraping utilities, class extraction, time parsing, and odds formatters."""

    def test_extract_race_class_all_variations(self):
        # Grade takes precedence
        assert _extract_race_class("東京優駿(G1)", "G1") == "G1"
        assert _extract_race_class("有馬記念(GI)", "G1") == "G1"

        # Parenthesized classes
        assert _extract_race_class("クローバー賞(OP)", None) == "OP"
        assert _extract_race_class("BSN賞(L)", None) == "L"
        assert _extract_race_class("STV賞(3勝)", None) == "3勝"
        assert _extract_race_class("積丹特別(2勝)", None) == "2勝"
        assert _extract_race_class("ニセコ特別(1勝)", None) == "1勝"

        # Japanese full text classes
        assert _extract_race_class("3歳未勝利", None) == "未勝利"
        assert _extract_race_class("2歳新馬", None) == "新馬"
        assert _extract_race_class("3歳以上1勝クラス", None) == "1勝"
        assert _extract_race_class("3歳以上2勝クラス", None) == "2勝"
        assert _extract_race_class("3歳以上3勝クラス", None) == "3勝"
        assert _extract_race_class("障害4歳以上オープン", None) == "OP"

        # Edge cases
        assert _extract_race_class("", None) is None
        assert _extract_race_class(None, None) is None
        assert _extract_race_class("普通のレース名", None) is None

    def test_parse_time(self):
        # Standard format 1:34.5 -> 94.5 seconds
        assert _parse_time("1:34.5") == 94.5
        # Sprint format 58.2 -> 58.2 seconds
        assert _parse_time("58.2") == 58.2
        # Long distance 3:15.8 -> 195.8 seconds
        assert _parse_time("3:15.8") == 195.8
        # None / empty / invalid
        assert _parse_time(None) is None
        assert _parse_time("") is None
        assert _parse_time("取消") is None
        assert _parse_time("---") is None

    def test_parse_weight(self):
        # 480(+4) -> (480, 4)
        assert _parse_weight("480(+4)") == (480, 4)
        # 512(-2) -> (512, -2)
        assert _parse_weight("512(-2)") == (512, -2)
        # 490(0) -> (490, 0)
        assert _parse_weight("490(0)") == (490, 0)
        # 474(前計不) -> (474, None)
        assert _parse_weight("474(前計不)") == (474, None)
        # 計不 -> (None, None)
        assert _parse_weight("計不") == (None, None)
        assert _parse_weight("") == (None, None)
        assert _parse_weight(None) == (None, None)

    def test_parse_odds_text(self):
        assert _parse_odds_text("3.4") == 3.4
        assert _parse_odds_text("120.5") == 120.5
        assert _parse_odds_text("---") is None
        assert _parse_odds_text("取消") is None
        assert _parse_odds_text(None) is None

    def test_live_odds_fetching_netkeiba_id(self):
        """Test fetch_win_odds on a known finished race ID (e.g. 202601020301)."""
        odds = fetch_win_odds("202601020301")
        if odds:
            assert isinstance(odds, list)
            assert len(odds) >= 1
            for o in odds:
                assert "combination" in o
                assert "odds_value" in o
                assert isinstance(o["odds_value"], float)
                assert o["odds_value"] >= 1.0
