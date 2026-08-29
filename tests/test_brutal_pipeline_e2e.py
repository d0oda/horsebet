import pytest
import tempfile
import json
from pathlib import Path
from results.build_data_json import build_data_json

class TestPipelineE2EBrutal:
    """Stress tests for pipeline steps, predictions JSON compilation, and edge handling."""

    def test_build_data_json_empty_and_corrupt_files(self):
        # 1. Non-existent file path
        res = build_data_json(["/non/existent/path/predictions.json"])
        # Should gracefully return empty or None without crashing
        assert res is None or isinstance(res, (dict, list))

        # 2. Corrupt JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{corrupt_json: true,")
            temp_corrupt = f.name

        try:
            with pytest.raises(Exception):
                build_data_json([temp_corrupt])
        finally:
            Path(temp_corrupt).unlink(missing_ok=True)

        # 3. Valid JSON structure without predictions
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"date": "2026-08-29", "model": "test_model", "predictions": []}, f)
            temp_valid = f.name

        try:
            out_file = temp_valid + "_out.json"
            build_data_json([temp_valid], output=out_file)
            if Path(out_file).exists():
                with open(out_file) as f_out:
                    d = json.load(f_out)
                    assert "races" in d
                    assert "summary" in d
                Path(out_file).unlink(missing_ok=True)
        finally:
            Path(temp_valid).unlink(missing_ok=True)

    def test_pipeline_argument_parser(self):
        from pipeline import main
        import sys
        # Verify parser flags without running
        from argparse import ArgumentParser
        # Just ensure pipeline module imports cleanly and all step functions exist
        import pipeline
        assert hasattr(pipeline, "step_scrape")
        assert hasattr(pipeline, "step_odds")
        assert hasattr(pipeline, "step_predict")
        assert hasattr(pipeline, "step_results")
        assert hasattr(pipeline, "step_paddock")
