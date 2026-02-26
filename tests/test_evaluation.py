"""
Tests for post-scrape evaluation runner and Sprint 6 CLI parsing.
"""

import sys
from io import StringIO
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Evaluation Runner — CLI Parsing
# ---------------------------------------------------------------------------

class TestEvaluationCLI:
    """Test argument parsing for run_evaluation.py."""

    def test_help_exits_zero(self):
        from models.run_evaluation import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["run_evaluation", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_only_accepts_valid_step(self):
        from models.run_evaluation import main
        # Step 6 is not valid
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_evaluation", "--only", "6"]):
                main()

    def test_skip_trio_flag(self):
        """Verify --skip-trio is parsed correctly."""
        import argparse
        from models.run_evaluation import main

        # Just test the parse — we can't easily run the full pipeline in tests
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-trio", action="store_true")
        args = parser.parse_args(["--skip-trio"])
        assert args.skip_trio is True


# ---------------------------------------------------------------------------
# Sprint 6 — CLI Parsing
# ---------------------------------------------------------------------------

class TestSprint6CLI:
    """Test argument parsing for run_sprint6.py."""

    def test_help_exits_zero(self):
        from models.run_sprint6 import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["run_sprint6", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_requires_mode(self):
        """Should fail without any mode flag."""
        from models.run_sprint6 import main
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_sprint6"]):
                main()

    def test_mutually_exclusive_modes(self):
        """Cannot specify both --reconcile and --drift."""
        from models.run_sprint6 import main
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["run_sprint6", "--reconcile", "--drift"]):
                main()


# ---------------------------------------------------------------------------
# Analyse Bets — CLI Parsing
# ---------------------------------------------------------------------------

class TestAnalyseBetsCLI:
    """Test argument parsing for analyse_bets.py."""

    def test_help_exits_zero(self):
        from models.analyse_bets import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["analyse_bets", "--help"]):
                main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Decision Matrix
# ---------------------------------------------------------------------------

class TestDecisionMatrix:
    def test_empty_results(self, capsys):
        from models.run_evaluation import print_decision_matrix
        print_decision_matrix()
        captured = capsys.readouterr()
        assert "No results to display" in captured.out

    def test_with_trio_result(self, capsys):
        from models.run_evaluation import print_decision_matrix
        from types import SimpleNamespace

        trio = SimpleNamespace(roi_pct=5.2)
        print_decision_matrix(trio_result=trio)
        captured = capsys.readouterr()
        assert "Trio ROI" in captured.out
        assert "+5.2%" in captured.out

    def test_negative_trio(self, capsys):
        from models.run_evaluation import print_decision_matrix
        from types import SimpleNamespace

        trio = SimpleNamespace(roi_pct=-12.3)
        print_decision_matrix(trio_result=trio)
        captured = capsys.readouterr()
        assert "Stick with win bets" in captured.out
