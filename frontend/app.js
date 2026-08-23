const DATA_URL = './data/data.json';

// ── State ──
let raceData = null;

// ── DOM refs ──
const appRoot = document.getElementById('app');
const headerDate = document.getElementById('header-date');
const headerModel = document.getElementById('header-model');
const summaryGrid = document.getElementById('summary-grid');
const racesList = document.getElementById('races-list');

// ── Fetch data ──
async function loadData() {
  appRoot.innerHTML = `
    <div class="container">
      <div class="loading-state">
        <div class="spinner"></div>
        <p style="color:var(--text-muted);font-size:13px;">Loading race data…</p>
      </div>
    </div>`;

  try {
    const res = await fetch(DATA_URL + '?t=' + Date.now());
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    raceData = await res.json();
    render();
  } catch (e) {
    showError(e);
  }
}

// ── Compute summary stats ──
function computeStats(races) {
  let totalBets = 0, wonBets = 0;
  let totalRaces = races.length;
  let racesWithBets = 0;
  let totalEvSum = 0, evCount = 0;

  for (const race of races) {
    const betEntry = race.entries.find(e => e.is_bet);
    if (betEntry) {
      totalBets++;
      racesWithBets++;
      if (betEntry.is_winner) wonBets++;
    }
    for (const e of race.entries) {
      if (e.ev !== null && e.ev !== undefined) {
        totalEvSum += e.ev;
        evCount++;
      }
    }
  }

  // Estimate P&L: assume ¥1000/bet, JRA pays odds×¥100 = payout
  // simplified: stake=1000, return=odds*1000 if win, else 0
  let pnl = 0;
  for (const race of races) {
    const betEntry = race.entries.find(e => e.is_bet);
    if (betEntry) {
      const stake = 1000;
      pnl -= stake;
      if (betEntry.is_winner) pnl += betEntry.odds * stake;
    }
  }

  return { totalRaces, totalBets, wonBets, pnl };
}

// ── Main render ──
function render() {
  const d = raceData;
  const races = d.races || [];
  const stats = computeStats(races);
  const dateStr = d.date ? formatDate(d.date) : '—';
  const modelStr = d.model || '—';
  const hasBetRaces = races.filter(r => r.has_bet);

  // Header
  headerDate.textContent = dateStr;
  headerModel.textContent = modelStr;

  // Re-render app content
  appRoot.innerHTML = `
    <div class="container">
      <div class="summary-grid" id="summary-grid"></div>
      <div class="section-header">
        <span class="section-title">Races</span>
        <span class="section-count">${races.length} races</span>
        ${hasBetRaces.length > 0 ? `<span class="section-count" style="background:var(--accent-dim);color:#a5b4fc;border-color:rgba(99,102,241,0.2)">${hasBetRaces.length} bets placed</span>` : ''}
      </div>
      <div class="races-list" id="races-list"></div>
    </div>`;

  // Summary stats
  const summaryEl = document.getElementById('summary-grid');
  const winRate = stats.totalBets > 0 ? Math.round(stats.wonBets / stats.totalBets * 100) : null;
  const pnlPositive = stats.pnl > 0;

  summaryEl.innerHTML = `
    ${statCard('RACES TODAY', races.length, '', 'stat-accent')}
    ${statCard('BETS PLACED', stats.totalBets, stats.totalBets === 0 ? 'No value found today' : '', 'stat-neutral')}
    ${stats.totalBets > 0 ? statCard('BETS WON', stats.wonBets, winRate !== null ? `${winRate}% win rate` : '', stats.wonBets > 0 ? 'stat-positive' : 'stat-negative') : ''}
    ${stats.totalBets > 0 ? statCard('P&L', formatPnl(stats.pnl), '¥1,000 / bet assumed', pnlPositive ? 'stat-positive' : 'stat-negative') : ''}
    ${statCard('STRATEGY', '—', d.strategy || '', 'stat-neutral')}
  `.replace(/^\s*$/gm, '');

  // Race list
  const racesEl = document.getElementById('races-list');
  if (races.length === 0) {
    racesEl.innerHTML = `<div class="empty-state"><div class="empty-icon">🏇</div><div class="empty-title">No races found</div><div class="empty-msg">No race data for this run yet.</div></div>`;
    return;
  }

  races.forEach((race, idx) => {
    const card = buildRaceCard(race, idx);
    racesEl.appendChild(card);
  });

  // Auto-open races with bets, or first race
  const cards = racesEl.querySelectorAll('.race-card');
  if (hasBetRaces.length > 0) {
    // Open only bet races
    cards.forEach(card => {
      if (card.classList.contains('has-bet')) toggleRace(card);
    });
  } else {
    // Open first race by default
    if (cards.length > 0) toggleRace(cards[0]);
  }
}

function statCard(label, value, sub, cls) {
  if (value === undefined || value === null || value === '') return '';
  return `
    <div class="stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${cls}">${value}</div>
      ${sub ? `<div class="stat-sub">${sub}</div>` : ''}
    </div>`;
}

// ── Build a race card ──
function buildRaceCard(race, idx) {
  const card = document.createElement('div');
  card.className = 'race-card' + (race.has_bet ? ' has-bet' : '');

  const surface = race.surface || 'turf';
  const surfaceTag = surface === 'dirt'
    ? `<span class="tag tag-dirt">Dirt</span>`
    : `<span class="tag tag-turf">Turf</span>`;

  const betTag = race.has_bet
    ? `<span class="tag tag-bet">📌 BET</span>`
    : `<span class="tag tag-nobets" style="opacity:0.4">No bet</span>`;

  const distStr = race.distance ? `${race.distance}m` : '';
  const goingStr = race.going || '';
  const venueStr = race.venue || '';

  // Sort entries by prob_combined desc for display (but keep finish_pos as result)
  const entries = [...(race.entries || [])].sort((a, b) => (b.prob_combined || 0) - (a.prob_combined || 0));

  const betEntry = entries.find(e => e.is_bet);
  const winner = entries.find(e => e.is_winner);

  // Bet result bar
  let betResultHtml = '';
  if (betEntry) {
    if (betEntry.is_winner) {
      const ret = (betEntry.odds * 1000).toLocaleString();
      betResultHtml = `<div class="bet-result-bar win">✅ BET WON — ${betEntry.horse_name} @${betEntry.odds}x → +¥${ret} return on ¥1,000</div>`;
    } else {
      const finishStr = betEntry.finish_pos !== null && betEntry.finish_pos !== undefined ? `Finished ${ordinal(betEntry.finish_pos)}` : 'DNF';
      betResultHtml = `<div class="bet-result-bar loss">❌ BET LOST — ${betEntry.horse_name} @${betEntry.odds}x — ${finishStr}${winner ? ` · Won by ${winner.horse_name}` : ''}</div>`;
    }
  }

  card.innerHTML = `
    <div class="race-header">
      <div class="race-num">${race.race_number}</div>
      <div class="race-title-block">
        <div class="race-name">${escHtml(race.race_name || 'Race')}</div>
        <div class="race-meta">
          <span class="race-meta-item">📍 ${escHtml(venueStr)}</span>
          ${distStr ? `<span class="race-meta-item">📏 ${distStr}</span>` : ''}
          ${goingStr ? `<span class="race-meta-item">🌧 ${escHtml(goingStr)}</span>` : ''}
        </div>
      </div>
      <div class="race-tags">
        ${surfaceTag}
        ${betTag}
      </div>
      <span class="race-chevron">›</span>
    </div>
    <div class="race-body" style="display:none">
      ${betResultHtml}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Horse</th>
              <th class="col-prob">Model%</th>
              <th class="col-prob-mkt">Mkt%</th>
              <th>Odds</th>
              <th>EV</th>
              <th class="col-last3f">Last 3F</th>
              <th>Finish</th>
              <th>Bet?</th>
            </tr>
          </thead>
          <tbody>
            ${entries.map(e => buildEntryRow(e)).join('')}
          </tbody>
        </table>
      </div>
    </div>`;

  card.querySelector('.race-header').addEventListener('click', () => toggleRace(card));
  return card;
}

// ── Build entry row ──
function buildEntryRow(e) {
  const isWinner = e.is_winner;
  const isBet = e.is_bet;
  const cls = [isWinner ? 'is-winner' : '', isBet ? 'is-bet' : ''].filter(Boolean).join(' ');

  const ev = e.ev;
  let evClass = 'ev-zero', evStr = '—';
  if (ev !== null && ev !== undefined) {
    evStr = (ev > 0 ? '+' : '') + ev.toFixed(1) + '%';
    evClass = ev > 0 ? 'ev-positive' : ev < -5 ? 'ev-negative' : 'ev-zero';
  }

  const prob = e.prob_combined;
  const probStr = prob !== null && prob !== undefined ? prob.toFixed(1) + '%' : '—';
  const probPct = Math.min(100, Math.max(0, prob || 0));

  const mktProb = e.prob_mkt;
  const mktStr = mktProb !== null && mktProb !== undefined ? mktProb.toFixed(1) + '%' : '—';

  const odds = e.odds !== null && e.odds !== undefined ? e.odds.toFixed(1) + 'x' : '—';

  const pos = e.finish_pos;
  let finishHtml;
  if (isWinner) {
    finishHtml = `<span class="finish-badge finish-1">🥇 1st</span>`;
  } else if (pos === null || pos === undefined) {
    finishHtml = `<span class="finish-badge finish-dnf">DNF</span>`;
  } else if (pos === 2) {
    finishHtml = `<span class="finish-badge finish-2">2nd</span>`;
  } else if (pos === 3) {
    finishHtml = `<span class="finish-badge finish-3">3rd</span>`;
  } else {
    finishHtml = `<span class="finish-badge finish-other">${pos}th</span>`;
  }

  const last3f = e.last_3f !== null && e.last_3f !== undefined ? e.last_3f.toFixed(1) + 's' : '—';

  const draw = e.draw || e.post_position || '?';
  const betHtml = isBet
    ? `<span class="bet-marker">📌</span>`
    : '';

  return `
    <tr class="${cls}">
      <td><span class="pos-num">${draw}</span></td>
      <td>
        <div class="horse-name">${escHtml(e.horse_name || '—')}</div>
        <div class="horse-meta">Pop. ${e.popularity ?? '?'}</div>
      </td>
      <td class="col-prob">
        <div class="prob-bar-wrap">
          <div class="prob-bar"><div class="prob-bar-fill" style="width:${probPct}%"></div></div>
          <span class="prob-text">${probStr}</span>
        </div>
      </td>
      <td class="col-prob-mkt">${mktStr}</td>
      <td style="font-weight:600">${odds}</td>
      <td class="${evClass}">${evStr}</td>
      <td class="col-last3f" style="color:var(--text-muted)">${last3f}</td>
      <td>${finishHtml}</td>
      <td>${betHtml}</td>
    </tr>`;
}

// ── Toggle race open/close ──
function toggleRace(card) {
  const body = card.querySelector('.race-body');
  const isOpen = card.classList.contains('open');
  card.classList.toggle('open', !isOpen);
  body.style.display = isOpen ? 'none' : 'block';
}

// ── Error state ──
function showError(e) {
  appRoot.innerHTML = `
    <div class="container">
      <div class="empty-state">
        <div class="empty-icon">🏇</div>
        <div class="empty-title">No race data yet</div>
        <div class="empty-msg">
          The GitHub Action hasn't run today, or data hasn't been published yet.<br><br>
          <small style="color:var(--text-dim)">${escHtml(String(e))}</small>
        </div>
      </div>
    </div>`;
}

// ── Helpers ──
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(d) {
  try {
    return new Date(d + 'T00:00:00').toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });
  } catch { return d; }
}

function formatPnl(pnl) {
  const sign = pnl >= 0 ? '+' : '';
  return sign + '¥' + Math.abs(Math.round(pnl)).toLocaleString();
}

function ordinal(n) {
  if (n === 1) return '1st';
  if (n === 2) return '2nd';
  if (n === 3) return '3rd';
  return n + 'th';
}

// ── Start ──
loadData();
