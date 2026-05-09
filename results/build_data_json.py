#!/usr/bin/env python3
"""
Build results/data.json from predictions JSON + database.

Usage:
    python results/build_data_json.py results/predictions_*.json
"""
import argparse
import json
import sys
import os
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

def build_data_json(predictions_paths: list, output: str = None):
    all_races = []
    all_value_bets = []
    global_total_bets = 0
    global_total_winners = 0
    global_total_returns = 0
    model_name = 'retrain_20260507_1645'
    dates = []

    for path in predictions_paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            preds = json.load(f)

        date = preds.get('date')
        if not date:
            continue
            
        dates.append(date)
        model_name = preds.get('model', model_name)
        pred_map = {p['entry_id']: p for p in preds['predictions']}

        with get_session() as session:
            races = session.execute(text('''
                SELECT r.id, r.race_name_jp, r.race_number, r.distance, r.surface,
                       r.going, r.course_id, r.date
                FROM horsebet.races r
                WHERE r.date = :date
                ORDER BY r.course_id, r.race_number
            '''), {'date': date}).fetchall()

            entries_db = session.execute(text('''
                SELECT e.id, e.race_id, e.post_position, e.draw, e.odds_win,
                       e.popularity, h.name_jp,
                       e.finish_pos, e.time_secs, e.last_3f_secs
                FROM horsebet.entries e
                JOIN horsebet.horses h ON h.id = e.horse_id
                WHERE e.race_id IN (SELECT id FROM horsebet.races WHERE date = :date)
                ORDER BY e.race_id, e.post_position
            '''), {'date': date}).fetchall()

        entry_map = {}
        for e in entries_db:
            entry_map.setdefault(e.race_id, []).append(e)

        for r in races:
            venue = COURSE_NAMES.get(r.course_id, f'Course_{r.course_id}')
            # We add date to the venue tab label so user can distinguish days
            date_str = r.date.strftime('%m-%d') if hasattr(r.date, 'strftime') else str(r.date)[5:]
            venue_with_date = f"{venue} ({date_str})" # e.g. Nakayama (03-07)
            
            race_entries = entry_map.get(r.id, [])
            entries_out = []
            for e in race_entries:
                p = pred_map.get(e.id, {})
                finish = e.finish_pos
                
                # Keep original data
                entries_out.append({
                    'horse_name': e.name_jp or '?',
                    'post_position': e.post_position,
                    'draw': e.draw,
                    'odds': e.odds_win or 0,
                    'popularity': e.popularity or 0,
                    'prob_combined': round((p.get('combined_win_prob', p.get('combined_prob', 0))) * 100, 1),
                    'prob_fund': round((p.get('model_win_prob', p.get('fundamental_prob', 0))) * 100, 1),
                    'prob_mkt': round((p.get('pace_win_prob', p.get('market_model_prob', 0))) * 100, 1),
                    'ev': round((p.get('ev', 0)) * 100, 1),
                    'finish_pos': finish,
                    'time_secs': e.time_secs,
                    'last_3f': e.last_3f_secs,
                    'is_winner': finish == 1 if finish else False,
                    'is_bet': False,
                    '_is_value': p.get('is_value_bet', p.get('is_value', False)),
                })

            entries_out.sort(key=lambda x: x['prob_combined'], reverse=True)

            value_entries = [e for e in entries_out if e['_is_value']]
            has_bet = False
            if value_entries:
                best = max(value_entries, key=lambda e: e['ev'])
                best['is_bet'] = True
                has_bet = True
                global_total_bets += 1

                is_winner = best.get('finish_pos') == 1
                if is_winner:
                    global_total_winners += 1
                    global_total_returns += int(best['odds'] * 1000)

                all_value_bets.append({
                    'venue': venue_with_date,
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

            for e in entries_out:
                if '_is_value' in e:
                    del e['_is_value']

            all_races.append({
                'race_number': r.race_number,
                'race_name': r.race_name_jp or '',
                'venue': venue_with_date,
                'distance': r.distance,
                'surface': r.surface or 'turf',
                'going': r.going or '',
                'has_bet': has_bet,
                'entries': entries_out,
                'date': str(r.date)
            })

    total_stake = global_total_bets * 1000
    display_date = " & ".join(sorted(list(set(dates)))) if dates else "Unknown Dates"
    
    data = {
        'model': model_name,
        'date': display_date,
        'strategy': 'max 1 bet/race, EV > threshold',
        'races': all_races,
        'summary': {
            'total_races': len(all_races),
            'total_entries': sum(len(r['entries']) for r in all_races),
            'total_bets': global_total_bets,
            'total_winners': global_total_winners,
            'strike_rate': round(global_total_winners / global_total_bets * 100, 1) if global_total_bets else 0,
            'total_stake': total_stake,
            'total_returns': global_total_returns,
            'roi': round((global_total_returns - total_stake) / total_stake * 100, 1) if total_stake else 0,
            'strategy': 'max 1 bet/race, EV > threshold',
        },
        'value_bets': all_value_bets
    }

    out_path = output or 'results/data.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    races_with = sum(1 for r in data['races'] if r['has_bet'])
    summary_text = f"✅ {out_path}: {len(all_races)} races, {global_total_bets} bets " \
                   f"({races_with} races w/ bet, {len(all_races) - races_with} skipped)\n"
    if global_total_winners:
        summary_text += f"   Winners: {global_total_winners}/{global_total_bets} ({data['summary']['strike_rate']}%)\n"
        summary_text += f"   ROI: {data['summary']['roi']}%\n"
        
    print(summary_text.strip())

    # Write human-readable bets.txt
    bets_txt_path = 'results/bets.txt'
    with open(bets_txt_path, 'w') as f:
        f.write(f"🏇 UmaEdge Value Bets ({display_date})\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Bets: {global_total_bets}\n")
        if global_total_winners:
            f.write(f"Winners: {global_total_winners}/{global_total_bets} ({data['summary']['strike_rate']}%)\n")
            f.write(f"ROI: {data['summary']['roi']}%\n")
        f.write("=" * 40 + "\n\n")

        # Sort races by venue and race number
        sorted_races = sorted(all_races, key=lambda x: (x['venue'], x['race_number']))
        if not sorted_races:
            f.write("No predictions found.\n")
        else:
            for race in sorted_races:
                f.write(f"📍 {race['venue']} R{race['race_number']} - {race['race_name']}\n")
                # Sort entries by EV descending
                sorted_entries = sorted(race['entries'], key=lambda x: x['ev'], reverse=True)
                for e in sorted_entries:
                    ev_sign = "+" if e['ev'] > 0 else ""
                    f.write(f"   🐴 #{e['post_position']} {e['horse_name']}\n")
                    f.write(f"      Odds: {e['odds']:.1f}x | EV: {ev_sign}{e['ev']:.1f}%\n")
                    f.write(f"      Win Prob: {e['prob_combined']:.1f}%\n")
                    if e.get('finish_pos') is not None:
                        res_str = '🏆 WINNER' if e['is_winner'] else f"Finished {e['finish_pos']}"
                        f.write(f"      Result: {res_str}\n")
                f.write("\n")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(f"### 📊 Pipeline Results ({display_date})\n")
            f.write(f"- **Races:** {len(all_races)} (Bets in {races_with})\n")
            f.write(f"- **Total Bets:** {global_total_bets}\n")
            if global_total_winners:
                f.write(f"- **Winners:** {global_total_winners} ({data['summary']['strike_rate']}%)\n")
                f.write(f"- **ROI:** {data['summary']['roi']}%\n")
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build data.json for frontend')
    parser.add_argument('predictions', nargs='+', help='Path to one or more predictions JSON files')
    parser.add_argument('--output', '-o', default='results/data.json', help='Output path')
    args = parser.parse_args()
    build_data_json(args.predictions, args.output)
