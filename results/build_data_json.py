#!/usr/bin/env python3
"""
Build results/data.json from predictions JSON + database.

Usage:
    python results/build_data_json.py results/march1_predictions.json
    python results/build_data_json.py results/march1_predictions.json --date 2026-03-01

Rules:
    - Max 1 bet per race
    - Only bet when horse is flagged as value bet (EV > threshold) by predict_final
    - Among value bets in a race, pick the one with highest EV
    - Races with no value bet → no bet (skip)
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.db import get_session
from sqlalchemy import text

COURSE_NAMES = {
    1: 'Sapporo', 2: 'Hakodate', 3: 'Fukushima', 4: 'Niigata',
    5: 'Tokyo', 6: 'Nakayama', 7: 'Chukyo', 8: 'Kyoto',
    9: 'Hanshin', 10: 'Kokura',
}


def build_data_json(predictions_path: str, date: str = None, output: str = None):
    with open(predictions_path) as f:
        preds = json.load(f)

    # Infer date from predictions if not given
    if not date:
        date = preds.get('date')
    if not date:
        raise ValueError("Date not found in predictions, specify --date")

    pred_map = {p['entry_id']: p for p in preds['predictions']}

    with get_session() as session:
        races = session.execute(text('''
            SELECT r.id, r.race_name_jp, r.race_number, r.distance, r.surface,
                   r.going, r.course_id
            FROM horsebet.races r
            WHERE r.date = :date
            ORDER BY r.course_id, r.race_number
        '''), {'date': date}).fetchall()

        entries_db = session.execute(text('''
            SELECT e.id, e.race_id, e.post_position, e.draw, e.odds_win,
                   e.popularity, h.name_jp
            FROM horsebet.entries e
            JOIN horsebet.horses h ON h.id = e.horse_id
            WHERE e.race_id IN (SELECT id FROM horsebet.races WHERE date = :date)
            ORDER BY e.race_id, e.post_position
        '''), {'date': date}).fetchall()

        # Also fetch results if available
        results_db = session.execute(text('''
            SELECT res.entry_id, res.finish_pos, res.time_secs, res.last_3f_secs
            FROM horsebet.results res
            JOIN horsebet.entries e ON e.id = res.entry_id
            WHERE e.race_id IN (SELECT id FROM horsebet.races WHERE date = :date)
        '''), {'date': date}).fetchall()

    result_map = {r.entry_id: r for r in results_db}
    entry_map = {}
    for e in entries_db:
        entry_map.setdefault(e.race_id, []).append(e)

    data = {
        'model': preds.get('model', '2026_v2'),
        'date': date,
        'strategy': 'max 1 bet/race, EV > threshold',
        'races': [],
    }

    value_bets_list = []
    total_bets = 0
    total_winners = 0
    total_returns = 0

    for r in races:
        venue = COURSE_NAMES.get(r.course_id, f'Course_{r.course_id}')
        race_entries = entry_map.get(r.id, [])

        entries_out = []
        for e in race_entries:
            p = pred_map.get(e.id, {})
            res = result_map.get(e.id)

            finish = res.finish_pos if res else None
            time_secs = res.time_secs if res else None
            last_3f = res.last_3f_secs if res else None

            entries_out.append({
                'horse_name': e.name_jp or '?',
                'post_position': e.post_position,
                'draw': e.draw,
                'odds': e.odds_win or 0,
                'popularity': e.popularity or 0,
                'prob_combined': round((p.get('combined_prob', 0)) * 100, 1),
                'prob_fund': round((p.get('fundamental_prob', 0)) * 100, 1),
                'prob_mkt': round((p.get('market_model_prob', 0)) * 100, 1),
                'ev': round((p.get('ev', 0)) * 100, 1),
                'finish_pos': finish,
                'time_secs': time_secs,
                'last_3f': last_3f,
                'is_winner': finish == 1 if finish else False,
                'is_bet': False,
                '_is_value': p.get('is_value_bet', False),
            })

        entries_out.sort(key=lambda x: x['prob_combined'], reverse=True)

        # Max 1 bet per race, only if value bet exists
        value_entries = [e for e in entries_out if e['_is_value']]
        has_bet = False
        if value_entries:
            best = max(value_entries, key=lambda e: e['ev'])
            best['is_bet'] = True
            has_bet = True
            total_bets += 1

            is_winner = best.get('finish_pos') == 1
            if is_winner:
                total_winners += 1
                total_returns += int(best['odds'] * 100)

            value_bets_list.append({
                'venue': venue,
                'race_number': r.race_number,
                'race_name': r.race_name_jp or '',
                'horse_name': best['horse_name'],
                'post_position': best['post_position'],
                'odds': best['odds'],
                'popularity': best['popularity'],
                'prob_combined': best['prob_combined'],
                'prob_fund': best['prob_fund'],
                'prob_mkt': best['prob_mkt'],
                'ev': best['ev'],
                'finish_pos': best.get('finish_pos'),
                'is_winner': is_winner,
            })

        # Clean internal fields
        for e in entries_out:
            del e['_is_value']

        data['races'].append({
            'race_number': r.race_number,
            'race_name': r.race_name_jp or '',
            'venue': venue,
            'distance': r.distance,
            'surface': r.surface or 'turf',
            'going': r.going or '',
            'has_bet': has_bet,
            'entries': entries_out,
        })

    total_stake = total_bets * 100
    data['summary'] = {
        'total_races': len(data['races']),
        'total_entries': sum(len(r['entries']) for r in data['races']),
        'total_bets': total_bets,
        'total_winners': total_winners,
        'strike_rate': round(total_winners / total_bets * 100, 1) if total_bets else 0,
        'total_stake': total_stake,
        'total_returns': total_returns,
        'roi': round((total_returns - total_stake) / total_stake * 100, 1) if total_stake else 0,
        'strategy': 'max 1 bet/race, EV > threshold',
    }
    data['value_bets'] = value_bets_list

    out_path = output or 'results/data.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    races_with = sum(1 for r in data['races'] if r['has_bet'])
    print(f"✅ {out_path}: {len(data['races'])} races, {total_bets} bets "
          f"({races_with} races w/ bet, {len(data['races']) - races_with} skipped)")
    if total_winners:
        print(f"   Winners: {total_winners}/{total_bets} ({data['summary']['strike_rate']}%)")
        print(f"   ROI: {data['summary']['roi']}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build data.json for frontend')
    parser.add_argument('predictions', help='Path to predictions JSON from predict_final')
    parser.add_argument('--date', help='Race date (YYYY-MM-DD), inferred from predictions if omitted')
    parser.add_argument('--output', '-o', default='results/data.json', help='Output path')
    args = parser.parse_args()
    build_data_json(args.predictions, args.date, args.output)
