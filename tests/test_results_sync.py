"""
Unit tests verifying synchronization between entries and results tables.
"""

import pytest
from sqlalchemy import text
from scraper.db import get_session
from scraper.netkeiba import RaceData, EntryData, HorseData, save_race_to_db


def test_save_race_synchronizes_entries_and_results():
    """Verify that saving finished race entries writes finish_pos and times to entries table as well."""
    test_netkeiba_id = "999901010101"
    race = RaceData(
        netkeiba_id=test_netkeiba_id,
        race_name_jp="テスト同期レース",
        date="2026-08-23",
        course_code="01",
        race_number=1,
        distance=1200,
        surface="芝",
        going="良",
        entries=[
            EntryData(
                post_position=1,
                draw=1,
                horse=HorseData(name="Test Horse 1", name_jp="テスト馬1", sex="牡", birth_year=2022, netkeiba_id="test_horse_sync_1"),
                jockey_name_jp="テスト騎手",
                weight_carried=55.0,
                odds_win=2.5,
                popularity=1,
                finish_pos=1,
                time_secs=70.5,
                last_3f_secs=34.2,
                corner_positions="1-1",
                margin="",
            ),
            EntryData(
                post_position=2,
                draw=2,
                horse=HorseData(name="Test Horse 2", name_jp="テスト馬2", sex="牝", birth_year=2022, netkeiba_id="test_horse_sync_2"),
                jockey_name_jp="テスト騎手2",
                weight_carried=55.0,
                odds_win=4.0,
                popularity=2,
                finish_pos=2,
                time_secs=70.8,
                last_3f_secs=34.5,
                corner_positions="2-2",
                margin="1 1/2",
            ),
        ],
    )

    try:
        # Save race
        success = save_race_to_db(race)
        assert success is True

        with get_session() as s:
            # Query entries and results
            rows = s.execute(text("""
                SELECT e.post_position, e.finish_pos, e.time_secs, e.last_3f_secs,
                       res.finish_pos as res_finish_pos, res.time_secs as res_time_secs
                FROM entries e
                JOIN races r ON r.id = e.race_id
                LEFT JOIN results res ON res.entry_id = e.id
                WHERE r.netkeiba_id = :nk_id
                ORDER BY e.post_position
            """), {"nk_id": test_netkeiba_id}).fetchall()

            assert len(rows) == 2
            # Entry 1
            assert rows[0].finish_pos == 1
            assert rows[0].res_finish_pos == 1
            assert rows[0].time_secs == 70.5
            assert rows[0].last_3f_secs == 34.2

            # Entry 2
            assert rows[1].finish_pos == 2
            assert rows[1].res_finish_pos == 2
            assert rows[1].time_secs == 70.8
            assert rows[1].last_3f_secs == 34.5

    finally:
        # Clean up test data
        with get_session() as s:
            race_id = s.execute(
                text("SELECT id FROM races WHERE netkeiba_id = :nk_id"),
                {"nk_id": test_netkeiba_id},
            ).scalar()
            if race_id:
                s.execute(text("DELETE FROM results WHERE entry_id IN (SELECT id FROM entries WHERE race_id = :rid)"), {"rid": race_id})
                s.execute(text("DELETE FROM entries WHERE race_id = :rid"), {"rid": race_id})
                s.execute(text("DELETE FROM races WHERE id = :rid"), {"rid": race_id})
                s.commit()
