// ── API Base URL ──
// Auto-detect: local dev → localhost:8000, production → Render
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : 'https://umaedge-api.onrender.com';

// ── State ──
let manifest = [];           // [{ date, races, bets, winners, strike_rate }]
let selectedDate = null;     // 'YYYY-MM-DD'
let calYear, calMonth;       // currently displayed month
let openRaceIds = new Set(); // track which race cards are open
let raceDetailCache = {};    // race_id → full detail (entries + predictions)

// ── DOM refs ──
const headerDate    = document.getElementById('header-date');
const headerModel   = document.getElementById('header-model');
const headerStatus  = document.getElementById('header-status');
const calGrid       = document.getElementById('cal-grid');
const calMonthLabel = document.getElementById('cal-month-label');
const calPrev       = document.getElementById('cal-prev');
const calNext       = document.getElementById('cal-next');
const racePanelInner = document.getElementById('race-panel-inner');
const manifestSummary = document.getElementById('manifest-summary');
const manifestList    = document.getElementById('manifest-list');

// ── Init ──
async function init() {
  const now = new Date();
  calYear  = now.getFullYear();
  calMonth = now.getMonth();

  calPrev.addEventListener('click', () => { calMonth--; if (calMonth < 0) { calMonth = 11; calYear--; } renderCalendar(); });
  calNext.addEventListener('click', () => { calMonth++; if (calMonth > 11) { calMonth = 0; calYear++; } renderCalendar(); });

  setStatus('loading');
  try {
    const res = await fetch(`${API_BASE}/api/manifest`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    manifest = data.dates || [];
    setStatus('live');
  } catch (e) {
    console.warn('Manifest fetch failed:', e);
    setStatus('offline');
  }

  renderCalendar();
  renderManifestList();

  // Auto-select today if it has data
  const todayStr = toDateStr(new Date());
  if (manifest.some(d => d.date === todayStr)) {
    selectDate(todayStr);
  }
}

// ── Status badge ──
function setStatus(state) {
  headerStatus.className = 'badge';
  if (state === 'live') {
    headerStatus.className = 'badge badge-live';
    headerStatus.textContent = '● Live';
  } else if (state === 'loading') {
    headerStatus.className = 'badge badge-loading';
    headerStatus.textContent = '◌ Loading…';
  } else {
    headerStatus.className = 'badge badge-date';
    headerStatus.textContent = '○ Offline';
  }
}

// ── Calendar ──
function renderCalendar() {
  const monthNames = ['January','February','March','April','May','June',
                      'July','August','September','October','November','December'];
  calMonthLabel.textContent = `${monthNames[calMonth]} ${calYear}`;

  const availableDates = new Set(manifest.map(d => d.date));
  const todayStr = toDateStr(new Date());

  // First day of the month (0=Sun)
  const firstDow = new Date(calYear, calMonth, 1).getDay();
  // Days in this month
  const daysInMonth = new Date(calYear, calMonth + 1, 0).getDate();
  // Days in previous month
  const daysInPrev = new Date(calYear, calMonth, 0).getDate();

  const cells = [];

  // Pad with previous month's days
  for (let i = firstDow - 1; i >= 0; i--) {
    cells.push({ day: daysInPrev - i, thisMonth: false, dow: firstDow - 1 - i });
  }

  // This month's days
  for (let d = 1; d <= daysInMonth; d++) {
    const dow = (firstDow + d - 1) % 7;
    cells.push({ day: d, thisMonth: true, dow });
  }

  // Pad to complete the last row
  const remaining = cells.length % 7 === 0 ? 0 : 7 - (cells.length % 7);
  for (let d = 1; d <= remaining; d++) {
    cells.push({ day: d, thisMonth: false, dow: (cells.length) % 7 });
  }

  calGrid.innerHTML = '';

  cells.forEach(cell => {
    const el = document.createElement('div');
    el.className = 'cal-day';

    if (!cell.thisMonth) {
      el.classList.add('other-month');
    } else {
      const dateStr = `${calYear}-${String(calMonth + 1).padStart(2,'0')}-${String(cell.day).padStart(2,'0')}`;
      const isWeekend = cell.dow === 0 || cell.dow === 6;
      const hasData = availableDates.has(dateStr);
      const isToday = dateStr === todayStr;
      const isSelected = dateStr === selectedDate;

      if (isWeekend)  el.classList.add('weekend');
      if (hasData)    el.classList.add('has-data');
      if (isToday)    el.classList.add('is-today');
      if (isSelected) el.classList.add('selected');

      el.setAttribute('data-date', dateStr);
      el.setAttribute('title', dateStr);

      // ALL this-month dates are clickable
      el.addEventListener('click', () => selectDate(dateStr));

      if (hasData) {
        const dot = document.createElement('span');
        dot.className = 'cal-dot';
        el.appendChild(dot);
      }
    }

    el.appendChild(document.createTextNode(cell.day));
    calGrid.appendChild(el);
  });
}

// ── Manifest sidebar list ──
function renderManifestList() {
  if (!manifest.length) return;
  manifestSummary.style.display = 'block';
  manifestList.innerHTML = '';

  // Sort: dates with bets first (most bets first), then most-recent
  const sorted = [...manifest].sort((a, b) => {
    if ((b.bets || 0) !== (a.bets || 0)) return (b.bets || 0) - (a.bets || 0);
    return b.date.localeCompare(a.date);
  });

  // Show up to 12 dates — enough to cover recent bet days
  const recent = sorted.slice(0, 12);
  recent.forEach(entry => {
    const row = document.createElement('div');
    row.className = 'manifest-row' + (entry.date === selectedDate ? ' selected' : '');
    row.dataset.date = entry.date;

    const d = new Date(entry.date + 'T00:00:00');
    const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const dayName = d.toLocaleDateString('en-US', { weekday: 'short' });

    row.innerHTML = `
      <div class="manifest-date">${dayName} ${label}</div>
      <div class="manifest-meta">${entry.races}R</div>
      ${entry.bets > 0 ? `<div class="manifest-bets">${entry.bets}B</div>` : ''}
    `;
    row.addEventListener('click', () => selectDate(entry.date));
    manifestList.appendChild(row);
  });
}

// ── Select a date ──
async function selectDate(dateStr) {
  selectedDate = dateStr;
  openRaceIds = new Set();
  raceDetailCache = {};

  // Update calendar + header
  renderCalendar();
  renderManifestList();

  const d = new Date(dateStr + 'T00:00:00');
  headerDate.textContent = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

  // Navigate calendar to show this month if needed
  if (d.getFullYear() !== calYear || d.getMonth() !== calMonth) {
    calYear  = d.getFullYear();
    calMonth = d.getMonth();
    renderCalendar();
  }

  // Always try to load races from the API — manifest is just a cache hint
  racePanelInner.innerHTML = `<div class="loading-state"><div class="spinner"></div><p style="color:var(--text-muted);font-size:12px">Checking ${dateStr}…</p></div>`;

  let races = [];
  try {
    const res = await fetch(`${API_BASE}/api/races?date=${dateStr}&limit=100`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    races = (await res.json()).races || [];
  } catch (e) {
    racePanelInner.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">API error</div>
        <div class="empty-msg">${escHtml(String(e))}<br><br>Is the API server running at ${API_BASE}?</div>
      </div>`;
    return;
  }

  const manifestEntry = manifest.find(m => m.date === dateStr);

  if (races.length === 0) {
    // No races in DB for this date — offer to fetch
    renderNoRacesState(dateStr);
  } else {
    // Races exist — render them (some may have predictions, some may not)
    renderRacePanel(races, manifestEntry || { date: dateStr, races: races.length, bets: 0, winners: 0 }, dateStr);
  }
}

// ── State: no races in DB yet for this date ──
function renderNoRacesState(dateStr) {
  const todayStr = toDateStr(new Date());
  const isFuture = dateStr > todayStr;
  const isRaceDay = (() => { const dow = new Date(dateStr + 'T00:00:00').getDay(); return dow === 0 || dow === 6; })();

  racePanelInner.innerHTML = `
    <div class="fetch-state">
      <div class="fetch-icon">🏇</div>
      <div class="fetch-title">No races fetched for ${formatDate(dateStr)}</div>
      <div class="fetch-msg">
        ${isFuture
          ? 'Races for this date haven’t been scraped yet. Click below to fetch them now.'
          : 'This date has no race data in the database. You can try scraping it manually.'}
      </div>
      <button class="action-btn" id="btn-fetch-${dateStr}" onclick="fetchRacesForDate('${dateStr}')">  
        📥 Fetch Races for ${dateStr}
      </button>
      ${!isRaceDay ? '<div class="fetch-hint">Note: JRA races are typically held on Sat &amp; Sun only.</div>' : ''}
    </div>`;
}

// ── Fetch races for a date via the scraper API ──
async function fetchRacesForDate(dateStr) {
  const btn = document.getElementById(`btn-fetch-${dateStr}`);
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Fetching… (this may take 30-60s)'; }

  racePanelInner.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <p style="color:var(--text-muted);font-size:13px;margin-bottom:6px">Scraping races for ${dateStr}…</p>
      <p style="color:var(--text-dim);font-size:11px">Fetching race cards from netkeiba. This usually takes 30–60 seconds.</p>
    </div>`;

  try {
    const res = await fetch(`${API_BASE}/api/scraper/date/${dateStr}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    // Reload races after scrape
    const racesRes = await fetch(`${API_BASE}/api/races?date=${dateStr}&limit=100`);
    const racesData = await racesRes.json();
    const races = racesData.races || [];

    if (races.length === 0) {
      racePanelInner.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🤔</div>
          <div class="empty-title">No races found</div>
          <div class="empty-msg">The scraper ran but found no races for ${dateStr}.<br>JRA may not have published the race card yet.</div>
        </div>`;
    } else {
      renderRacePanel(races, { date: dateStr, races: races.length, bets: 0, winners: 0 }, dateStr);
    }
  } catch (e) {
    racePanelInner.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">❌</div>
        <div class="empty-title">Scrape failed</div>
        <div class="empty-msg">${escHtml(String(e))}</div>
        <button class="action-btn" onclick="fetchRacesForDate('${dateStr}')">Retry</button>
      </div>`;
  }
}

// ── Run predictions for one race ──
async function runPredictionsForRace(raceId, dateStr) {
  const btn = document.getElementById(`btn-predict-${raceId}`);
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Predicting…'; }

  try {
    const res = await fetch(`${API_BASE}/api/races/${raceId}/refresh`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    // Invalidate cache and re-render this card
    delete raceDetailCache[raceId];
    const card = document.querySelector(`[data-race-id="${raceId}"]`);
    if (card && card.classList.contains('open')) {
      const body = document.getElementById(`race-body-${raceId}`);
      body.innerHTML = `<div class="race-body-loading"><div class="spinner" style="width:18px;height:18px;border-width:2px;margin:0"></div><span>Reloading predictions…</span></div>`;
      const detailRes = await fetch(`${API_BASE}/api/races/${raceId}`);
      raceDetailCache[raceId] = await detailRes.json();
      renderRaceDetail(body, raceDetailCache[raceId], { id: raceId });
    }
    // Update the button to show done
    const newBtn = document.getElementById(`btn-predict-${raceId}`);
    if (newBtn) {
      const isRerun = newBtn.classList.contains('predict-btn-rerun');
      if (isRerun) {
        // Re-run: briefly show done, then reset to ↺ (re-runnable)
        newBtn.textContent = `✅ Done`;
        newBtn.disabled = true;
        setTimeout(() => {
          newBtn.textContent = '↺';
          newBtn.disabled = false;
        }, 2500);
      } else {
        // Fresh prediction: show done count, disable permanently (↺ will appear on reload)
        newBtn.textContent = `✅ Done (${data.predictions || '?'} preds)`;
        newBtn.disabled = true;
      }
    }
  } catch (e) {
    const errBtn = document.getElementById(`btn-predict-${raceId}`);
    if (errBtn) { errBtn.disabled = false; errBtn.textContent = errBtn.classList.contains('predict-btn-rerun') ? '↺' : `❌ Failed — retry`; errBtn.title = String(e); }
  }
}

// ── Render the race panel for a date ──
function renderRacePanel(races, manifestEntry, dateStr) {
  const bets = manifestEntry.bets || 0;
  const winners = manifestEntry.winners || 0;
  const strikeRate = manifestEntry.strike_rate;
  const totalRaces = manifestEntry.races || races.length;

  // Sort races by course_id, then race_number
  races.sort((a, b) => (a.course_id - b.course_id) || (a.race_number - b.race_number));

  // Group by venue
  const venues = {};
  races.forEach(r => {
    const v = r.course_name || `Course ${r.course_id}`;
    (venues[v] = venues[v] || []).push(r);
  });

  const venueList = Object.entries(venues);
  const hasBetRaces = races.filter(r => r._has_bet);

  let html = `
    <div class="summary-grid">
      ${statCard('RACES', totalRaces, '', 'stat-accent')}
      ${statCard('BETS PLACED', bets, bets === 0 ? 'No value bets found' : '', 'stat-neutral')}
      ${bets > 0 ? statCard('WINNERS', winners, strikeRate !== null ? `${strikeRate}% strike rate` : '', winners > 0 ? 'stat-positive' : 'stat-negative') : ''}
      ${statCard('VENUES', venueList.length, '', 'stat-neutral')}
    </div>
  `;

  if (races.length === 0) {
    html += `<div class="empty-state"><div class="empty-icon">🏇</div><div class="empty-title">No races found</div></div>`;
    racePanelInner.innerHTML = html;
    return;
  }

  // Section header
  html += `
    <div class="section-header">
      <span class="section-title">Races — ${formatDate(dateStr)}</span>
      <span class="section-count">${races.length} races</span>
    </div>
    <div class="races-list" id="races-list"></div>`;

  racePanelInner.innerHTML = html;

  // Build race cards
  const racesList = document.getElementById('races-list');
  races.forEach(race => {
    racesList.appendChild(buildRaceCard(race));
  });
}

// ── Build a race card (shows bet status immediately from race list data) ──
function buildRaceCard(race) {
  const card = document.createElement('div');
  card.className = 'race-card' + (race.has_bet ? ' has-bet' : '');
  card.dataset.raceId = race.id;

  const surface = (race.surface || 'turf').toLowerCase();
  const surfaceTag = surface === 'dirt'
    ? `<span class="tag tag-dirt">Dirt</span>`
    : `<span class="tag tag-turf">Turf</span>`;

  const dist     = race.distance ? `${race.distance}m` : '';
  const venue    = race.course_name || '';
  const postTime = race.post_time ? race.post_time.slice(0, 5) : '';

  // Bet tag — visible immediately from race list data
  const betTag = race.has_bet
    ? `<span class="tag tag-bet">📌 BET</span>`
    : race.has_predictions
      ? `<span class="tag tag-predicted" style="opacity:0.55">Predicted</span>`
      : `<span class="tag tag-nobets" style="opacity:0.5">No bet</span>`;

  // Bet preview line (horse name + odds) shown in header if available
  const betPreview = race.has_bet && race.bet_horse
    ? `<span class="bet-preview">#${race.bet_post_position} ${escHtml(race.bet_horse)} @${(race.bet_odds||0).toFixed(1)}x &nbsp;<span class="bet-ev">${race.bet_ev > 0 ? '+' : ''}${(race.bet_ev * 100).toFixed(0)}% EV</span></span>`
    : '';

  // Always allow re-running predictions (odds change throughout the day).
  // Show full "▶ Predict" when no predictions exist yet; subtle "↺" re-run icon otherwise.
  const predictBtn = !race.has_predictions
    ? `<button class="predict-btn" id="btn-predict-${race.id}" onclick="event.stopPropagation(); runPredictionsForRace(${race.id}, '${race.date}')">▶ Predict</button>`
    : `<button class="predict-btn predict-btn-rerun" id="btn-predict-${race.id}" onclick="event.stopPropagation(); runPredictionsForRace(${race.id}, '${race.date}')" title="Re-run with latest odds">↺</button>`;

  card.innerHTML = `
    <div class="race-header" id="race-header-${race.id}">
      <div class="race-num${race.has_bet ? ' race-num-bet' : ''}">${race.race_number}</div>
      <div class="race-title-block">
        <div class="race-name">${escHtml(race.race_name_jp || race.race_name || 'Race')}</div>
        <div class="race-meta">
          ${venue    ? `<span>📍 ${escHtml(venue)}</span>` : ''}
          ${dist     ? `<span>📏 ${dist}</span>`           : ''}
          ${postTime ? `<span>🕐 ${postTime}</span>`            : ''}
          ${betPreview}
        </div>
      </div>
      <div class="race-tags" id="race-tags-${race.id}">${surfaceTag}${betTag}</div>
      ${predictBtn}
      <span class="race-chevron">›</span>
    </div>
    <div class="race-body" id="race-body-${race.id}"></div>`;

  card.querySelector('.race-header').addEventListener('click', () => toggleRace(card, race));
  return card;
}

// ── Toggle race open/close (lazy-loads detail) ──
async function toggleRace(card, race) {
  const isOpen = card.classList.contains('open');
  card.classList.toggle('open', !isOpen);
  const body = document.getElementById(`race-body-${race.id}`);

  if (!isOpen) {
    // Opening — load detail if not cached
    if (!raceDetailCache[race.id]) {
      body.innerHTML = `<div class="race-body-loading"><div class="spinner" style="width:18px;height:18px;border-width:2px;margin:0"></div><span>Loading entries…</span></div>`;
      try {
        const res = await fetch(`${API_BASE}/api/races/${race.id}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        raceDetailCache[race.id] = await res.json();
      } catch (e) {
        body.innerHTML = `<div class="race-body-loading" style="color:var(--red)">⚠️ Failed to load: ${escHtml(String(e))}</div>`;
        return;
      }
    }
    renderRaceDetail(body, raceDetailCache[race.id], race);

    // Update card classes based on bets
    const hasBet = raceDetailCache[race.id]?.value_bets?.length > 0;
    // Primary bet = highest EV value bet
    const sortedVB = [...(raceDetailCache[race.id]?.value_bets || [])].sort((a,b) => (b.ev||0)-(a.ev||0));
    const hasPrimaryBet = sortedVB.length > 0;
    card.classList.toggle('has-bet', hasPrimaryBet);
    const num = card.querySelector('.race-num');
    if (hasPrimaryBet) num.style.cssText = 'background:var(--accent-dim);color:#a5b4fc;border:1px solid rgba(99,102,241,0.25)';
    const hasPreds = (raceDetailCache[race.id]?.predictions?.length || 0) > 0;
    const tags = card.querySelector('.race-tags');
    tags.innerHTML = (race.surface === 'dirt'
      ? `<span class="tag tag-dirt">Dirt</span>`
      : `<span class="tag tag-turf">Turf</span>`)
      + (hasPrimaryBet
          ? `<span class="tag tag-bet">📌 BET</span>`
          : hasPreds
            ? `<span class="tag tag-predicted" style="opacity:0.55">Predicted</span>`
            : `<span class="tag tag-nobets" style="opacity:0.4">No bet</span>`);
  }
}

// ── Render race body with entries + predictions ──
function renderRaceDetail(body, detail, race) {
  const entries = detail.entries || [];
  const predictions = detail.predictions || [];
  const valueBets = detail.value_bets || [];

  // Build prediction map by entry_id
  const predMap = {};
  predictions.forEach(p => { predMap[p.entry_id] = p; });

  // Sort value bets by EV desc — pick the single best one as THE bet
  const sortedValueBets = [...valueBets].sort((a, b) => (b.ev || 0) - (a.ev || 0));
  const primaryBetEntryId = sortedValueBets.length > 0 ? sortedValueBets[0].entry_id : null;
  const otherValueEntryIds = new Set(sortedValueBets.slice(1).map(v => v.entry_id));

  // Build value bet EV map keyed by entry_id
  const vbEvMap = {};
  valueBets.forEach(vb => { vbEvMap[vb.entry_id] = vb.ev; });

  // Build unified entry list
  const unified = entries.map(e => {
    const pred = predMap[e.id] || {};
    const isPrimaryBet = e.id === primaryBetEntryId;  // THE bet — highest EV
    const isValueOnly  = otherValueEntryIds.has(e.id); // clears threshold but not the top pick
    // Use value_bet EV when pred.edge is null (older predictions may not have edge stored)
    const vbEv = vbEvMap[e.id] ?? null;
    return { ...e, pred, isPrimaryBet, isValueOnly, vbEv };
  });

  // Sort: by win_prob desc if available, else post_position
  unified.sort((a, b) => {
    const pa = a.pred.win_prob || 0;
    const pb = b.pred.win_prob || 0;
    if (pa !== pb) return pb - pa;
    return (a.post_position || 0) - (b.post_position || 0);
  });

  // Bet result bar (only for the primary bet)
  const betEntry = unified.find(e => e.isPrimaryBet);
  let betResultHtml = '';
  if (betEntry) {
    if (betEntry.finish_pos === 1) {
      const ret = ((betEntry.odds_win || 0) * 1000).toLocaleString();
      betResultHtml = `<div class="bet-result-bar win">✅ BET WON — ${escHtml(betEntry.horse_name_jp || '?')} @${betEntry.odds_win}x → +¥${ret}</div>`;
    } else if (betEntry.finish_pos) {
      const winner = unified.find(e => e.finish_pos === 1);
      betResultHtml = `<div class="bet-result-bar loss">❌ BET LOST — ${escHtml(betEntry.horse_name_jp || '?')} @${betEntry.odds_win}x — Finished ${ordinal(betEntry.finish_pos)}${winner ? ` · Won by ${escHtml(winner.horse_name_jp || '?')}` : ''}</div>`;
    }
  }

  // Build model header label
  const modelVersion = predictions[0]?.model_version || '';

  body.innerHTML = `
    ${betResultHtml}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Horse</th>
            <th>Model%</th>
            <th class="col-mkt">Mkt%</th>
            <th>Odds</th>
            <th>EV</th>
            <th class="col-last3f">L3F</th>
            <th>Finish</th>
            <th>Bet?</th>
          </tr>
        </thead>
        <tbody>
          ${unified.map(e => buildEntryRow(e)).join('')}
        </tbody>
      </table>
    </div>`;

  if (modelVersion) {
    headerModel.textContent = modelVersion;
  }
}

// ── Build entry table row ──
function buildEntryRow(e) {
  const isWinner    = e.finish_pos === 1;
  const isPrimaryBet = e.isPrimaryBet;  // THE single bet for this race
  const isValueOnly  = e.isValueOnly;   // clears threshold but not the top pick
  const pred        = e.pred || {};

  const cls = [isWinner ? 'is-winner' : '', isPrimaryBet ? 'is-bet' : ''].filter(Boolean).join(' ');

  // Win probability
  const prob = pred.win_prob != null ? pred.win_prob * 100 : null;
  const probStr = prob != null ? prob.toFixed(1) + '%' : '—';
  const probPct = Math.min(100, Math.max(0, prob || 0));

  // Market prob from odds
  const odds = e.odds_win;
  const mktProb = odds && odds > 1 ? (1 / odds * 100) : null;
  const mktStr  = mktProb != null ? mktProb.toFixed(1) + '%' : '—';
  const oddsStr = odds != null ? odds.toFixed(1) + 'x' : '—';

  // EV — prefer pred.edge, fall back to value_bet ev (for older predictions)
  const edge = pred.edge != null ? parseFloat(pred.edge) : (isPrimaryBet || isValueOnly ? e.vbEv : null);
  let evClass = 'ev-zero', evStr = '—';
  if (edge != null) {
    evStr  = (edge > 0 ? '+' : '') + (edge * 100).toFixed(1) + '%';
    evClass = edge > 0 ? 'ev-positive' : edge < -0.05 ? 'ev-negative' : 'ev-zero';
  }

  // Finish
  const pos = e.finish_pos;
  let finishHtml;
  if (isWinner)         finishHtml = `<span class="finish-badge finish-1">🥇 1st</span>`;
  else if (!pos)        finishHtml = `<span class="finish-badge finish-dnf">—</span>`;
  else if (pos === 2)   finishHtml = `<span class="finish-badge finish-2">2nd</span>`;
  else if (pos === 3)   finishHtml = `<span class="finish-badge finish-3">3rd</span>`;
  else                  finishHtml = `<span class="finish-badge finish-other">${pos}th</span>`;

  const last3f = e.last_3f_secs != null ? e.last_3f_secs.toFixed(1) + 's' : '—';
  const pp     = e.post_position || '?';
  // 📌 = the one actual bet; ◆ VALUE = clears threshold but not top pick
  const betHtml = isPrimaryBet
    ? `<span class="bet-marker">📌 BET</span>`
    : isValueOnly
      ? `<span class="bet-marker value-only" title="Clears EV threshold but not the top pick">◆ VALUE</span>`
      : '';

  return `
    <tr class="${cls}">
      <td><span class="pos-num">${pp}</span></td>
      <td>
        <div class="horse-name">${escHtml(e.horse_name_jp || e.horse_name || '—')}</div>
        <div class="horse-meta">Pop. ${e.popularity ?? '?'}</div>
      </td>
      <td>
        <div class="prob-bar-wrap">
          <div class="prob-bar"><div class="prob-bar-fill" style="width:${probPct}%"></div></div>
          <span class="prob-text">${probStr}</span>
        </div>
      </td>
      <td class="col-mkt">${mktStr}</td>
      <td style="font-weight:600">${oddsStr}</td>
      <td class="${evClass}">${evStr}</td>
      <td class="col-last3f" style="color:var(--text-muted)">${last3f}</td>
      <td>${finishHtml}</td>
      <td>${betHtml}</td>
    </tr>`;
}

// ── Helpers ──
function statCard(label, value, sub, cls) {
  if (value === undefined || value === null || value === '') return '';
  return `
    <div class="stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${cls}">${value}</div>
      ${sub ? `<div class="stat-sub">${sub}</div>` : ''}
    </div>`;
}

function toDateStr(d) {
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

function formatDate(dateStr) {
  try {
    return new Date(dateStr + 'T00:00:00').toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });
  } catch { return dateStr; }
}

function ordinal(n) {
  if (n === 1) return '1st';
  if (n === 2) return '2nd';
  if (n === 3) return '3rd';
  return n + 'th';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Start ──
init();
