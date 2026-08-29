// ── API Base URL ──
// Auto-detect: if served directly by backend on port 8000 or Render → relative '', if served by static server on port 3000 → http://localhost:8000
const API_BASE = (window.location.port === '8000' || window.location.origin.includes('onrender.com'))
  ? ''
  : ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://umaedge-api.onrender.com');

const round1 = v => (v == null || isNaN(v)) ? 0.0 : Math.round(v * 10) / 10;

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

  renderCalendar();
  setStatus('loading');
  try {
    const res = await fetch(`${API_BASE}/api/manifest`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    manifest = data.dates || [];
    if (data.model_version && headerModel) {
      headerModel.textContent = data.model_version;
    }
    setStatus('live');
  } catch (e) {
    console.warn('Manifest fetch failed:', e);
    setStatus('offline');
  }

  renderCalendar();
  renderManifestList();

  // Auto-select today if it has data, or the latest available date
  const todayStr = toDateStr(new Date());
  if (manifest.some(d => d.date === todayStr)) {
    await selectDate(todayStr);
  } else if (manifest.length > 0) {
    const sorted = [...manifest].sort((a, b) => b.date.localeCompare(a.date));
    await selectDate(sorted[0].date);
  }
}

// ── Quick Date Jumpers ──
function jumpToUpcomingDate() {
  const todayStr = toDateStr(new Date());
  const upcomingInManifest = manifest.filter(m => m.date >= todayStr).sort((a, b) => a.date.localeCompare(b.date));
  if (upcomingInManifest.length > 0) {
    selectDate(upcomingInManifest[0].date);
    return;
  }
  const now = new Date();
  const daysUntilSat = (6 - now.getDay() + 7) % 7 || 7;
  const nextSat = new Date(now.getTime() + daysUntilSat * 24 * 60 * 60 * 1000);
  selectDate(toDateStr(nextSat));
}

function jumpToToday() {
  const todayStr = toDateStr(new Date());
  selectDate(todayStr);
}

function jumpToLatestResults() {
  const todayStr = toDateStr(new Date());
  const completed = manifest.filter(m => m.date <= todayStr).sort((a, b) => b.date.localeCompare(a.date));
  if (completed.length > 0) {
    selectDate(completed[0].date);
  } else if (manifest.length > 0) {
    selectDate(manifest[0].date);
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

  // Sort chronologically: most recent dates first
  const sorted = [...manifest].sort((a, b) => b.date.localeCompare(a.date));

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

let activeSelectDateToken = 0;
// ── Select a date ──
async function selectDate(dateStr) {
  const currentToken = ++activeSelectDateToken;
  selectedDate = dateStr;
  openRaceIds = new Set();
  raceDetailCache = {};
  currentFilter = 'all';

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
    if (currentToken !== activeSelectDateToken) return;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    races = (await res.json()).races || [];
  } catch (e) {
    if (currentToken !== activeSelectDateToken) return;
    racePanelInner.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <div class="empty-title">API error</div>
        <div class="empty-msg">${escHtml(String(e))}<br><br>Is the API server running at ${API_BASE}?</div>
      </div>`;
    return;
  }

  if (currentToken !== activeSelectDateToken) return;
  const manifestEntry = manifest.find(m => m.date === dateStr);

  if (races.length === 0) {
    // No races in DB for this date — offer to fetch
    renderNoRacesState(dateStr);
  } else {
    // Races exist — render them (some may have predictions, some may not)
    await renderRacePanel(races, manifestEntry || { date: dateStr, races: races.length, bets: 0, winners: 0 }, dateStr);
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

// ── Fetch or refetch races for a date via the scraper API ──
async function fetchRacesForDate(dateStr, force = false) {
  const fetchBtn = document.getElementById(`btn-fetch-${dateStr}`);
  const refetchBtn = document.getElementById('btn-refetch-races');
  const bannerBtn = document.getElementById('btn-provisional-refetch');

  if (fetchBtn) {
    fetchBtn.disabled = true;
    fetchBtn.textContent = '⏳ Fetching… (this may take 30-60s)';
  }
  if (refetchBtn) {
    refetchBtn.disabled = true;
    refetchBtn.innerHTML = `<div class="spinner" style="width:13px;height:13px;border-width:2px;margin:0;display:inline-block;vertical-align:middle;"></div> <span>Refetching…</span>`;
  }
  if (bannerBtn) {
    bannerBtn.disabled = true;
    bannerBtn.innerHTML = `<div class="spinner" style="width:13px;height:13px;border-width:2px;margin:0;display:inline-block;vertical-align:middle;"></div> <span>Refetching card…</span>`;
  }

  const isInitialEmpty = !currentRaces || currentRaces.length === 0;
  if (isInitialEmpty) {
    racePanelInner.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p style="color:var(--text-muted);font-size:13px;margin-bottom:6px">Scraping races for ${dateStr}…</p>
        <p style="color:var(--text-dim);font-size:11px">Fetching race cards from netkeiba. This usually takes 30–60 seconds.</p>
      </div>`;
  } else {
    showToast(`🔄 Refetching race cards for ${formatDate(dateStr)} from netkeiba…`, 'info', 4000);
  }

  try {
    const url = `${API_BASE}/api/scraper/date/${dateStr}${force ? '?force=true' : ''}`;
    const res = await fetch(url, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    // Refresh manifest so calendar and sidebar counts update immediately
    try {
      const mRes = await fetch(`${API_BASE}/api/manifest`);
      if (mRes.ok) {
        manifest = (await mRes.json()).dates || [];
        renderCalendar();
        renderManifestList();
      }
    } catch (err) {
      console.warn('Failed to refresh manifest:', err);
    }

    // Clear detail cache for freshly scraped races
    raceDetailCache = {};

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
          <button class="action-btn" onclick="fetchRacesForDate('${dateStr}', true)">Retry</button>
        </div>`;
    } else {
      const manifestEntry = manifest.find(m => m.date === dateStr);
      showToast(`✨ Refetched race card: <strong>${races.length} races</strong> now available for ${formatDate(dateStr)}!`, 'success', 4500);
      renderRacePanel(races, manifestEntry || { date: dateStr, races: races.length, bets: 0, winners: 0 }, dateStr);
    }
  } catch (e) {
    if (isInitialEmpty) {
      racePanelInner.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">❌</div>
          <div class="empty-title">Scrape failed</div>
          <div class="empty-msg">${escHtml(String(e))}</div>
          <button class="action-btn" onclick="fetchRacesForDate('${dateStr}', true)">Retry</button>
        </div>`;
    } else {
      showToast(`❌ Scrape failed: ${escHtml(String(e))}`, 'error', 5000);
      if (refetchBtn) {
        refetchBtn.disabled = false;
        refetchBtn.innerHTML = `<span class="btn-icon">📥</span> Refetch Races`;
      }
      if (bannerBtn) {
        bannerBtn.disabled = false;
        bannerBtn.innerHTML = `<span class="btn-icon">🔄</span> Refetch Full Card`;
      }
    }
  }
}

// ── State variables for filtering and races ──
let currentRaces = [];
let currentFilter = 'all';

// ── Floating Toast Notification ──
function showToast(message, type = 'info', duration = 4500) {
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  toast.className = `toast-item toast-${type}`;
  toast.innerHTML = `
    <div class="toast-content">${message}</div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('toast-fadeout');
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

// ── Batch Predict All Races for Date ──
async function predictAllRacesForDate(dateStr) {
  const btn = document.getElementById('btn-predict-all');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner" style="width:14px;height:14px;border-width:2px;margin:0;display:inline-block;vertical-align:middle;"></div> <span>Predicting ${currentRaces.length} races with AI…</span>`;
  }

  showToast(`⚡ Running AI predictions for all races on ${formatDate(dateStr)}…`, 'info', 3500);

  try {
    const res = await fetch(`${API_BASE}/api/races/date/${dateStr}/predict?refresh_odds=true`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const worthCount = data.value_bets_count || 0;
    const racesCount = data.races_predicted || currentRaces.length;

    // Refresh manifest so calendar and sidebar counts update immediately
    try {
      const mRes = await fetch(`${API_BASE}/api/manifest`);
      if (mRes.ok) {
        manifest = (await mRes.json()).dates || [];
        renderCalendar();
        renderManifestList();
      }
    } catch (err) {
      console.warn('Failed to refresh manifest:', err);
    }

    // Clear detail cache for these races
    raceDetailCache = {};

    showToast(`✨ Successfully predicted ${racesCount} races! Found <strong>${worthCount} value bet${worthCount === 1 ? '' : 's'}</strong> 🔥`, 'success', 5000);

    // Reload the date to reflect updated predictions & bet status
    await selectDate(dateStr);

    // If there are value bets, trigger pulse animation on the value bet filter chip
    if (worthCount > 0) {
      const worthChip = document.getElementById('chip-filter-bets');
      if (worthChip) {
        worthChip.classList.add('pulse-highlight');
        setTimeout(() => worthChip.classList.remove('pulse-highlight'), 3000);
      }
    }
  } catch (e) {
    showToast(`❌ Batch prediction failed: ${escHtml(String(e.message || e))}`, 'error', 6000);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `⚠️ Retry Predict All`;
    }
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
        newBtn.textContent = `✅ Done`;
        newBtn.disabled = true;
        setTimeout(() => {
          newBtn.textContent = '↺';
          newBtn.disabled = false;
        }, 2500);
      } else {
        newBtn.textContent = `✅ Done (${data.predictions || '?'} preds)`;
        newBtn.disabled = true;
      }
    }

    // Refresh manifest to reflect any new bet
    try {
      const mRes = await fetch(`${API_BASE}/api/manifest`);
      if (mRes.ok) {
        manifest = (await mRes.json()).dates || [];
        renderCalendar();
        renderManifestList();
      }
    } catch (_) {}

  } catch (e) {
    const errBtn = document.getElementById(`btn-predict-${raceId}`);
    if (errBtn) { errBtn.disabled = false; errBtn.textContent = errBtn.classList.contains('predict-btn-rerun') ? '↺' : `❌ Failed — retry`; errBtn.title = String(e); }
  }
}

// ── Sync results for a date ──
async function syncResultsForDate(dateStr, force = false) {
  const btn = document.getElementById('btn-sync-results');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner" style="width:13px;height:13px;border-width:2px;margin:0;display:inline-block;vertical-align:middle;"></div> <span>Syncing results…</span>`;
  }
  showToast(`🏁 Syncing race results and payouts for ${formatDate(dateStr)}…`, 'info', 4000);

  try {
    const url = `${API_BASE}/api/results/date/${dateStr}${force ? '?force=true' : ''}`;
    const res = await fetch(url, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    // Refresh manifest
    try {
      const mRes = await fetch(`${API_BASE}/api/manifest`);
      if (mRes.ok) {
        manifest = (await mRes.json()).dates || [];
        renderCalendar();
        renderManifestList();
      }
    } catch (_) {}

    raceDetailCache = {};

    if (data.races_synced > 0) {
      showToast(`✨ Successfully recorded official results for ${data.races_synced} race${data.races_synced === 1 ? '' : 's'}!`, 'success', 4500);
    } else {
      const todayIso = toDateStr(new Date());
      if (dateStr >= todayIso) {
        showToast(`⏳ Races for ${formatDate(dateStr)} have not run yet (or are in progress). Official results are published by netkeiba after each race finishes.`, 'info', 5000);
      } else {
        showToast(`ℹ️ All races for ${formatDate(dateStr)} are already up-to-date with official results.`, 'info', 4000);
      }
    }
    await selectDate(dateStr);
  } catch (e) {
    showToast(`❌ Results sync failed: ${escHtml(String(e.message || e))}`, 'error', 5000);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span class="btn-icon">🏁</span> Sync Results`;
    }
  }
}

// ── Rescrape a single race card & re-predict ──
async function rescrapeSingleRace(raceId, dateStr) {
  const btn = document.getElementById(`btn-rescrape-${raceId}`);
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner" style="width:12px;height:12px;border-width:2px;margin:0;display:inline-block;vertical-align:middle;"></div> <span>Rescraping…</span>`;
  }
  showToast(`📥 Rescraping race card from netkeiba…`, 'info', 3500);

  try {
    const res = await fetch(`${API_BASE}/api/races/${raceId}/rescrape`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    delete raceDetailCache[raceId];
    const card = document.querySelector(`[data-race-id="${raceId}"]`);
    if (card && card.classList.contains('open')) {
      const body = document.getElementById(`race-body-${raceId}`);
      body.innerHTML = `<div class="race-body-loading"><div class="spinner" style="width:18px;height:18px;border-width:2px;margin:0"></div><span>Reloading updated card…</span></div>`;
      const detailRes = await fetch(`${API_BASE}/api/races/${raceId}`);
      raceDetailCache[raceId] = await detailRes.json();
      renderRaceDetail(body, raceDetailCache[raceId], { id: raceId });
    }

    showToast(`✨ Race rescraped and updated!`, 'success', 3500);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span class="btn-icon">📥</span> Rescrape`;
    }
  } catch (e) {
    showToast(`❌ Race rescrape failed: ${escHtml(String(e.message || e))}`, 'error', 5000);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span class="btn-icon">📥</span> Rescrape`;
    }
  }
}

let portfolioViewMode = 'daily'; // 'daily' | 'monthly'
let currentSelectedMonth = '2026-08';

async function switchPortfolioView(mode, month = null) {
  portfolioViewMode = mode;
  if (month) currentSelectedMonth = month;
  if (!currentSelectedMonth && selectedDate) {
    currentSelectedMonth = selectedDate.slice(0, 7);
  }
  if (portfolioViewMode === 'daily') {
    await loadDailyReturns(selectedDate);
  } else {
    await loadMonthlyReturns(currentSelectedMonth || (selectedDate ? selectedDate.slice(0, 7) : '2026-08'));
  }
}

// ── Load and render daily returns for all 4 strategies ──
async function loadDailyReturns(dateStr, force = false) {
  const container = document.getElementById('daily-returns-container');
  if (!container) return;

  if (portfolioViewMode === 'monthly') {
    return loadMonthlyReturns(currentSelectedMonth || (dateStr ? dateStr.slice(0, 7) : '2026-08'), force);
  }

  try {
    const res = await fetch(`${API_BASE}/api/races/date/${dateStr}/daily-returns?budget_per_race=1000${force ? '&force_refresh=true' : ''}`);
    if (!res.ok) {
      container.style.display = 'none';
      return;
    }
    const data = await res.json();
    if (!data || !data.strategies) {
      container.style.display = 'none';
      return;
    }

    const strategies = data.strategies;
    const hasResults = data.has_results;
    container.style.display = 'block';

    let html = `
      <div class="daily-returns-card">
        <div class="daily-returns-header">
          <div class="returns-title-group">
            <span class="returns-badge">📊 PORTFOLIO PERFORMANCE</span>
            <div class="returns-view-toggle">
              <button class="view-toggle-btn active" onclick="switchPortfolioView('daily')">📅 Daily</button>
              <button class="view-toggle-btn" onclick="switchPortfolioView('monthly')">🗓️ Monthly</button>
            </div>
            <span class="returns-title">Daily Returns & P&L — ${formatDate(dateStr)} (¥1,000 / race)</span>
          </div>
          <div class="returns-meta-info">
            ${hasResults ? `<span class="returns-status-live">🏁 Official Payouts Calculated</span>` : `<span class="returns-status-pending">⏳ Official Results Pending</span>`}
          </div>
        </div>

        <div class="strategy-return-grid">
    `;

    const stratKeys = ['auto', 'pure_win', 'hybrid', 'dutching'];
    stratKeys.forEach(k => {
      const s = strategies[k];
      if (!s) return;
      const isProfitable = s.profit > 0;
      const isZero = s.profit === 0;
      const profitClass = isZero ? 'return-profit-neutral' : isProfitable ? 'return-profit-positive' : 'return-profit-negative';
      const profitSign = s.profit > 0 ? '+' : '';

      html += `
        <div class="strategy-return-item ${s.id}">
          <div class="strategy-item-top">
            <div class="strategy-name-box">
              <span class="strategy-item-icon">${s.icon}</span>
              <div class="strategy-text-wrap">
                <div class="strategy-item-name">${escHtml(s.name)}</div>
                <div class="strategy-item-tagline">${escHtml(s.tagline)}</div>
              </div>
            </div>
            ${hasResults && isProfitable ? `<span class="roi-pill positive">+${s.roi_pct}% ROI</span>` : hasResults && !isZero ? `<span class="roi-pill negative">${s.roi_pct}% ROI</span>` : ''}
          </div>

          <div class="strategy-numbers-row">
            <div class="strat-stat">
              <span class="strat-stat-label">STAKED</span>
              <span class="strat-stat-val">¥${s.staked.toLocaleString()}</span>
            </div>
            <div class="strat-stat">
              <span class="strat-stat-label">PAYOUT</span>
              <span class="strat-stat-val ${isProfitable ? 'text-green' : ''}">¥${s.payout.toLocaleString()}</span>
            </div>
            <div class="strat-stat strat-stat-profit">
              <span class="strat-stat-label">NET P&L</span>
              <span class="strat-stat-val ${profitClass}">
                ${hasResults ? `${profitSign}¥${s.profit.toLocaleString()}` : '—'}
              </span>
            </div>
          </div>

          <div class="strategy-hit-row">
            <div class="hit-label">Hit Rate:</div>
            <div class="hit-value">${hasResults ? `<strong>${s.bets_won} / ${s.bets_placed}</strong> wins (${s.strike_rate}%)` : `<strong>${s.bets_placed}</strong> value bets active`}</div>
          </div>
        </div>
      `;
    });

    html += `
        </div>
      </div>
    `;

    container.innerHTML = html;
  } catch (e) {
    console.error('Failed to load daily returns:', e);
    container.style.display = 'none';
  }
}

// ── Load and render monthly returns across all 4 strategies ──
async function loadMonthlyReturns(monthStr, force = false) {
  const container = document.getElementById('daily-returns-container');
  if (!container) return;

  currentSelectedMonth = monthStr;

  try {
    const res = await fetch(`${API_BASE}/api/races/month/${monthStr}/monthly-returns?budget_per_race=1000${force ? '&force_refresh=true' : ''}`);
    if (!res.ok) {
      container.style.display = 'none';
      return;
    }
    const data = await res.json();
    if (!data || !data.strategies) {
      container.style.display = 'none';
      return;
    }

    const strategies = data.strategies;
    const hasResults = data.has_results;
    const availMonths = data.available_months || [monthStr];
    container.style.display = 'block';

    const monthChipsHtml = availMonths.map(m => {
      const [yr, mo] = m.split('-');
      const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const label = `${monthNames[parseInt(mo, 10)] || mo} ${yr}`;
      const isActive = m === monthStr;
      return `<button class="month-chip ${isActive ? 'active' : ''}" onclick="switchPortfolioView('monthly', '${m}')">${label}</button>`;
    }).join('');

    let html = `
      <div class="daily-returns-card monthly-returns-card">
        <div class="daily-returns-header">
          <div class="returns-title-group">
            <span class="returns-badge">📊 PORTFOLIO PERFORMANCE</span>
            <div class="returns-view-toggle">
              <button class="view-toggle-btn" onclick="switchPortfolioView('daily')">📅 Daily</button>
              <button class="view-toggle-btn active" onclick="switchPortfolioView('monthly')">🗓️ Monthly</button>
            </div>
            <span class="returns-title">Monthly Overview — ${escHtml(data.month_name || monthStr)} (${data.total_race_days} Race Days · ${data.total_races} Races)</span>
          </div>
          <div class="returns-meta-info">
            ${hasResults ? `<span class="returns-status-live">🏁 ${data.total_finished_races} Races Settled</span>` : `<span class="returns-status-pending">⏳ Official Results Pending</span>`}
          </div>
        </div>

        <!-- Month Navigation Bar -->
        <div class="month-picker-bar">
          <span class="month-picker-label">Select Month:</span>
          ${monthChipsHtml}
        </div>

        <div class="strategy-return-grid">
    `;

    const stratKeys = ['auto', 'pure_win', 'hybrid', 'dutching'];
    stratKeys.forEach(k => {
      const s = strategies[k];
      if (!s) return;
      const isProfitable = s.profit > 0;
      const isZero = s.profit === 0;
      const profitClass = isZero ? 'return-profit-neutral' : isProfitable ? 'return-profit-positive' : 'return-profit-negative';
      const profitSign = s.profit > 0 ? '+' : '';

      html += `
        <div class="strategy-return-item ${s.id}">
          <div class="strategy-item-top">
            <div class="strategy-name-box">
              <span class="strategy-item-icon">${s.icon}</span>
              <div class="strategy-text-wrap">
                <div class="strategy-item-name">${escHtml(s.name)}</div>
                <div class="strategy-item-tagline">${escHtml(s.tagline)}</div>
              </div>
            </div>
            ${hasResults && isProfitable ? `<span class="roi-pill positive">+${s.roi_pct}% ROI</span>` : hasResults && !isZero ? `<span class="roi-pill negative">${s.roi_pct}% ROI</span>` : ''}
          </div>

          <div class="strategy-numbers-row">
            <div class="strat-stat">
              <span class="strat-stat-label">STAKED</span>
              <span class="strat-stat-val">¥${s.staked.toLocaleString()}</span>
            </div>
            <div class="strat-stat">
              <span class="strat-stat-label">PAYOUT</span>
              <span class="strat-stat-val ${isProfitable ? 'text-green' : ''}">¥${s.payout.toLocaleString()}</span>
            </div>
            <div class="strat-stat strat-stat-profit">
              <span class="strat-stat-label">NET P&L</span>
              <span class="strat-stat-val ${profitClass}">
                ${hasResults ? `${profitSign}¥${s.profit.toLocaleString()}` : '—'}
              </span>
            </div>
          </div>

          <div class="strategy-hit-row">
            <div class="hit-label">Hit Rate:</div>
            <div class="hit-value">${hasResults ? `<strong>${s.bets_won} / ${s.bets_placed}</strong> wins (${s.strike_rate}%)` : `<strong>${s.bets_placed}</strong> value bets active`}</div>
          </div>
        </div>
      `;
    });

    html += `
        </div>

        <!-- Daily Breakdown Timeline Table -->
        <div class="monthly-breakdown-section">
          <div class="monthly-breakdown-header">
            <span class="monthly-breakdown-title">🗓️ Daily Performance Breakdown (${data.daily_timeline ? data.daily_timeline.length : 0} Race Days)</span>
            <span class="monthly-breakdown-sub">Click any race day row to view that date's racecard</span>
          </div>
          <div class="monthly-timeline-table-wrap">
            <table class="monthly-timeline-table">
              <thead>
                <tr>
                  <th>Race Date</th>
                  <th>Races</th>
                  <th>Value Bets</th>
                  <th>Staked</th>
                  <th>Payout</th>
                  <th>Net P&L</th>
                  <th>ROI</th>
                  <th>Hit Rate</th>
                  <th style="text-align:center;">Action</th>
                </tr>
              </thead>
              <tbody>
    `;

    const timeline = data.daily_timeline || [];
    timeline.forEach(row => {
      const isRowProfitable = row.profit > 0;
      const isRowZero = row.profit === 0;
      const pnlClass = isRowZero ? 'neutral' : isRowProfitable ? 'positive' : 'negative';
      const pnlSign = row.profit > 0 ? '+' : '';
      const isCurrentDay = row.date === selectedDate;

      html += `
        <tr class="${isCurrentDay ? 'active-day-row' : ''}" onclick="selectDate('${row.date}')" title="View race card for ${formatDate(row.date)}">
          <td><strong>${formatDate(row.date)}</strong> ${isCurrentDay ? '<span style="color:#a5b4fc;font-size:10px;margin-left:4px;">(Current)</span>' : ''}</td>
          <td>${row.total_races}</td>
          <td>${row.bets_placed}</td>
          <td>¥${row.staked.toLocaleString()}</td>
          <td class="${isRowProfitable ? 'text-green' : ''}">¥${row.payout.toLocaleString()}</td>
          <td>
            <span class="timeline-pnl-badge ${pnlClass}">
              ${pnlSign}¥${row.profit.toLocaleString()}
            </span>
          </td>
          <td class="${isRowProfitable ? 'text-green' : isRowZero ? '' : 'text-red'}">
            <strong>${row.roi_pct > 0 ? '+' : ''}${row.roi_pct}%</strong>
          </td>
          <td>${row.bets_won} / ${row.bets_placed} (${row.bets_placed > 0 ? ((row.bets_won/row.bets_placed)*100).toFixed(1) : 0}%)</td>
          <td style="text-align:center;">
            <button class="timeline-view-btn" onclick="event.stopPropagation(); selectDate('${row.date}')">View →</button>
          </td>
        </tr>
      `;
    });

    html += `
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = html;
  } catch (e) {
    console.error('Failed to load monthly returns:', e);
    container.style.display = 'none';
  }
}

// ── Filter races by type/venue/surface/winners ──
function setRaceFilter(filterName) {
  currentFilter = filterName;

  // Update active state on all chips
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.classList.remove('active');
    if (filterName === 'all' && chip.id === 'chip-filter-all') {
      chip.classList.add('active');
    } else if (filterName === 'bets' && chip.id === 'chip-filter-bets') {
      chip.classList.add('active');
    } else if (filterName === 'winners' && chip.id === 'chip-filter-winners') {
      chip.classList.add('active');
    } else if (filterName === 'turf' && chip.id === 'chip-filter-turf') {
      chip.classList.add('active');
    } else if (filterName === 'dirt' && chip.id === 'chip-filter-dirt') {
      chip.classList.add('active');
    } else if (chip.getAttribute('data-filter') === filterName) {
      chip.classList.add('active');
    }
  });

  renderRaceCardsList();
}

// ── Render list of race cards matching currentFilter ──
function renderRaceCardsList() {
  const racesList = document.getElementById('races-list');
  if (!racesList) return;
  racesList.innerHTML = '';

  let filtered = currentRaces;
  if (currentFilter === 'bets') {
    filtered = currentRaces.filter(r => r.has_bet);
  } else if (currentFilter === 'winners') {
    filtered = currentRaces.filter(r => r.bet_won);
  } else if (currentFilter === 'turf') {
    filtered = currentRaces.filter(r => (r.surface || '').toLowerCase() === 'turf' || (r.surface || '').includes('芝'));
  } else if (currentFilter === 'dirt') {
    filtered = currentRaces.filter(r => (r.surface || '').toLowerCase() === 'dirt' || (r.surface || '').includes('ダ'));
  } else if (currentFilter !== 'all') {
    filtered = currentRaces.filter(r => (r.course_name || `Course ${r.course_id}`) === currentFilter);
  }

  if (filtered.length === 0) {
    if (currentFilter === 'winners') {
      racesList.innerHTML = `
        <div class="empty-filter-state">
          <div class="empty-filter-icon">🏆</div>
          <div class="empty-filter-title">No Winning Bets on This Day</div>
          <div class="empty-filter-msg">
            ${currentRaces.some(r => r.has_bet)
              ? 'Races for this date have completed, but value bet selections did not finish in 1st place.'
              : 'Official race results are pending or no value bets were placed on this date.'}
          </div>
          <button class="filter-chip active" style="margin-top:14px;padding:8px 16px;cursor:pointer;" onclick="setRaceFilter('all')">
            ← View All ${currentRaces.length} Races
          </button>
        </div>
      `;
    } else if (currentFilter === 'bets') {
      racesList.innerHTML = `
        <div class="empty-filter-state">
          <div class="empty-filter-icon">🛡️</div>
          <div class="empty-filter-title">No Value Bets on This Day</div>
          <div class="empty-filter-msg">
            None of the ${currentRaces.length} races cleared the minimum Expected Value threshold (+8% EV).<br>
            All runners are currently priced fairly or underpriced by the market. Capital is safely preserved.
          </div>
          <button class="filter-chip active" style="margin-top:14px;padding:8px 16px;cursor:pointer;" onclick="setRaceFilter('all')">
            ← View All ${currentRaces.length} Races
          </button>
        </div>
      `;
    } else {
      racesList.innerHTML = `
        <div class="empty-filter-state">
          <div class="empty-filter-icon">🏇</div>
          <div class="empty-filter-title">No Races Match Filter</div>
          <button class="filter-chip active" style="margin-top:14px;padding:8px 16px;cursor:pointer;" onclick="setRaceFilter('all')">
            ← View All Races
          </button>
        </div>
      `;
    }
    return;
  }

  filtered.forEach(race => {
    racesList.appendChild(buildRaceCard(race));
  });
}

// ── Render the race panel for a date ──
async function renderRacePanel(races, manifestEntry, dateStr) {
  currentRaces = races;
  const bets = manifestEntry.bets || races.filter(r => r.has_bet).length;
  const wonCount = races.filter(r => r.bet_won).length;
  const winners = manifestEntry.winners != null ? manifestEntry.winners : wonCount;
  const strikeRate = manifestEntry.strike_rate != null ? manifestEntry.strike_rate : (bets > 0 ? round1((wonCount / bets) * 100) : null);
  const totalRaces = manifestEntry.races || races.length;
  const predictedRacesCount = races.filter(r => r.has_predictions).length;
  const unpredictedRacesCount = totalRaces - predictedRacesCount;
  const worthItCount = races.filter(r => r.has_bet).length;

  // Sort races by course_id, then race_number
  races.sort((a, b) => (a.course_id - b.course_id) || (a.race_number - b.race_number));

  // Group by venue
  const venues = {};
  races.forEach(r => {
    const v = r.course_name || `Course ${r.course_id}`;
    (venues[v] = venues[v] || []).push(r);
  });
  const venueList = Object.entries(venues);

  let html = `
    <div class="summary-grid">
      ${statCard('RACES', totalRaces, '', 'stat-accent')}
      ${statCard('VALUE BETS', worthItCount, worthItCount === 0 ? 'No positive EV edge' : `${worthItCount} value picks (click to filter)`, worthItCount > 0 ? 'stat-positive stat-clickable' : 'stat-neutral', 'onclick="setRaceFilter(\'bets\')"' )}
      ${bets > 0 ? statCard('WINNERS', winners, strikeRate !== null ? `${strikeRate}% strike rate (click to filter)` : (wonCount > 0 ? `${round1(wonCount/worthItCount*100)}% strike rate (click to filter)` : ''), (winners > 0 || wonCount > 0) ? 'stat-positive stat-clickable' : 'stat-negative', 'onclick="setRaceFilter(\'winners\')"' ) : ''}
      ${statCard('VENUES', venueList.length, '', 'stat-neutral')}
    </div>

    <!-- Daily Multi-Strategy Performance & Returns Card -->
    <div id="daily-returns-container" style="display:none;margin-bottom:18px;"></div>
  `;

  if (races.length === 0) {
    html += `<div class="empty-state"><div class="empty-icon">🏇</div><div class="empty-title">No races found</div></div>`;
    racePanelInner.innerHTML = html;
    return;
  }

  // Predict all button label & style
  let predictAllBtnHtml = '';
  if (predictedRacesCount === 0) {
    predictAllBtnHtml = `
      <button class="btn-predict-all primary" id="btn-predict-all" onclick="predictAllRacesForDate('${dateStr}')">
        <span class="btn-icon">⚡</span> Predict All Races (${totalRaces})
      </button>
    `;
  } else if (unpredictedRacesCount > 0) {
    predictAllBtnHtml = `
      <button class="btn-predict-all primary" id="btn-predict-all" onclick="predictAllRacesForDate('${dateStr}')">
        <span class="btn-icon">⚡</span> Predict All (${unpredictedRacesCount} new / ${totalRaces})
      </button>
    `;
  } else {
    predictAllBtnHtml = `
      <button class="btn-predict-all repredict" id="btn-predict-all" onclick="predictAllRacesForDate('${dateStr}')" title="Re-evaluate all races with current model & odds">
        <span class="btn-icon">↺</span> Repredict All Races (${totalRaces})
      </button>
    `;
  }

  // Provisional Race Notice (e.g. only 12 special races announced early in the week)
  let provisionalNoticeHtml = '';
  if (races.length < 24) {
    provisionalNoticeHtml = `
      <div class="provisional-card-banner">
        <span class="provisional-icon">ℹ️</span>
        <div class="provisional-content">
          <div class="provisional-header-row">
            <div class="provisional-title">Provisional Special Entries (${races.length} Races Announced)</div>
            <button class="provisional-action-btn" id="btn-provisional-refetch" onclick="fetchRacesForDate('${dateStr}', true)" title="Check netkeiba for updated or finalized full race card">
              <span class="btn-icon">🔄</span> Refetch Full Card
            </button>
          </div>
          <div class="provisional-desc">JRA has announced the special races (特別登録). The complete 36-race card (R1–R12) and official post positions will be drawn and finalized on Thursday (~16:00 JST). Live betting odds open Friday evening.</div>
        </div>
      </div>
    `;
  }

  // Section header & interactive toolbar
  html += `
    ${provisionalNoticeHtml}
    <div class="section-header">
      <div class="section-title-group">
        <span class="section-title">Races — ${formatDate(dateStr)}</span>
        <span class="section-count">${races.length} races ${worthItCount > 0 ? `· <strong style="color:#4ade80">${worthItCount} value bet${worthItCount === 1 ? '' : 's'}</strong>` : ''} ${wonCount > 0 ? `· <strong style="color:#fde047">🏆 ${wonCount} won</strong>` : ''}</span>
      </div>
      <div class="section-actions">
        ${(dateStr <= toDateStr(new Date())) ? `
          <button class="btn-sync-results" id="btn-sync-results" onclick="syncResultsForDate('${dateStr}')" title="Scrape and record official finish positions, payouts, and times">
            <span class="btn-icon">🏁</span> Sync Results
          </button>
        ` : ''}
        <button class="btn-refetch-races" id="btn-refetch-races" onclick="fetchRacesForDate('${dateStr}', true)" title="Scrape latest race cards & entries from netkeiba">
          <span class="btn-icon">📥</span> Refetch Races
        </button>
        ${predictAllBtnHtml}
      </div>
    </div>

    <!-- Raceday Filter Toolbar -->
    <div class="raceday-toolbar">
      <div class="filter-chips">
        <button class="filter-chip ${currentFilter === 'all' ? 'active' : ''}" id="chip-filter-all" onclick="setRaceFilter('all')">
          All Races <span class="chip-count">${totalRaces}</span>
        </button>
        <button class="filter-chip filter-chip-worth ${currentFilter === 'bets' ? 'active' : ''} ${worthItCount > 0 ? 'has-bets' : ''}" id="chip-filter-bets" onclick="setRaceFilter('bets')">
          <span class="chip-flame">🔥</span> Value Bets <span class="chip-count">${worthItCount}</span>
        </button>
        ${wonCount > 0 ? `
          <button class="filter-chip filter-chip-winners ${currentFilter === 'winners' ? 'active' : ''}" id="chip-filter-winners" onclick="setRaceFilter('winners')">
            <span class="chip-trophy">🏆</span> Winning Bets <span class="chip-count">${wonCount}</span>
          </button>
        ` : ''}
        <button class="filter-chip ${currentFilter === 'turf' ? 'active' : ''}" id="chip-filter-turf" onclick="setRaceFilter('turf')">
          🌱 Turf <span class="chip-count">${races.filter(r => (r.surface || '').toLowerCase() === 'turf' || (r.surface || '').includes('芝')).length}</span>
        </button>
        <button class="filter-chip ${currentFilter === 'dirt' ? 'active' : ''}" id="chip-filter-dirt" onclick="setRaceFilter('dirt')">
          🏜️ Dirt <span class="chip-count">${races.filter(r => (r.surface || '').toLowerCase() === 'dirt' || (r.surface || '').includes('ダ')).length}</span>
        </button>
        ${venueList.map(([vName, vRaces]) => `
          <button class="filter-chip ${currentFilter === vName ? 'active' : ''}" data-filter="${escHtml(vName)}" onclick="setRaceFilter('${escHtml(vName)}')">
            📍 ${escHtml(vName)} <span class="chip-count">${vRaces.length}</span>
          </button>
        `).join('')}
      </div>
    </div>

    <div class="races-list" id="races-list"></div>
  `;

  racePanelInner.innerHTML = html;
  renderRaceCardsList();
  loadDailyReturns(dateStr);
}

// ── Build a race card (shows bet status immediately from race list data) ──
function buildRaceCard(race) {
  const card = document.createElement('div');
  const isWon = Boolean(race.bet_won);
  const isWorthIt = Boolean(race.has_bet);
  card.className = 'race-card' + (isWon ? ' has-won-bet has-bet has-worth-bet' : isWorthIt ? ' has-bet has-worth-bet' : '');
  card.dataset.raceId = race.id;

  const surface = (race.surface || 'turf').toLowerCase();
  const surfaceTag = surface === 'dirt'
    ? `<span class="tag tag-dirt">Dirt</span>`
    : `<span class="tag tag-turf">Turf</span>`;

  const dist     = race.distance ? `${race.distance}m` : '';
  const venue    = race.course_name || '';
  const postTime = race.post_time ? race.post_time.slice(0, 5) : '';
  const isFinished = Boolean(race.is_finished || isWon || race.bet_finish_pos != null);
  const oddsTime = formatOddsTime(race.odds_updated_at || race.odds_as_of || race.pred_created_at);
  const oddsTagTitle = isFinished
    ? `Official closing SP odds locked at race post time (${escHtml(oddsTime)} JST). Predictions evaluated with official closing odds.`
    : `Live market odds captured as of ${escHtml(oddsTime)} JST`;
  const oddsTagLabel = isFinished ? `⏱ Final: ${escHtml(oddsTime)}` : `⏱ Odds: ${escHtml(oddsTime)}`;

  // Bet tag — visible immediately from race list data
  const betTag = isWon
    ? `<span class="tag tag-won"><span class="trophy-icon">🏆</span> BET WON</span>`
    : isWorthIt
      ? `<span class="tag tag-worth-bet"><span class="flame-icon">🔥</span> VALUE BET</span>`
      : race.has_predictions
        ? `<span class="tag tag-predicted" style="opacity:0.65">Predicted (Pass)</span>`
        : `<span class="tag tag-nobets" style="opacity:0.5">No bet</span>`;

  // Bet preview line (horse name + odds + EV or Payout) shown in header if available
  const betPreview = isWorthIt && race.bet_horse
    ? `<span class="bet-preview ${isWon ? 'won' : 'worth-it'}">
         <span class="bet-horse-num">#${race.bet_post_position}</span>
         <span class="bet-horse-name">${escHtml(race.bet_horse)}</span>
         <span class="bet-odds">@${(race.bet_odds||0).toFixed(1)}x</span>
         ${isWon
           ? `<span class="bet-won-pill">✅ 1st · +¥${(race.bet_payout || Math.round((race.bet_odds||0)*1000)).toLocaleString()}</span>`
           : `<span class="bet-ev">${race.bet_ev > 0 ? '+' : ''}${(race.bet_ev * 100).toFixed(0)}% EV</span>`}
       </span>`
    : '';

  // Always allow re-running predictions (odds change throughout the day).
  // Show full "▶ Predict" when no predictions exist yet; subtle "↺" re-run icon otherwise.
  const predictBtn = !race.has_predictions
    ? `<button class="predict-btn" id="btn-predict-${race.id}" onclick="event.stopPropagation(); runPredictionsForRace(${race.id}, '${race.date}')">▶ Predict</button>`
    : `<button class="predict-btn predict-btn-rerun" id="btn-predict-${race.id}" onclick="event.stopPropagation(); runPredictionsForRace(${race.id}, '${race.date}')" title="Re-run with latest odds (odds recorded at ${oddsTime})">↺</button>`;

  card.innerHTML = `
    <div class="race-header" id="race-header-${race.id}">
      <div class="race-num${isWon ? ' race-num-won' : isWorthIt ? ' race-num-bet' : ''}">${isWon ? '🏆 ' : ''}${race.race_number}</div>
      <div class="race-title-block">
        <div class="race-name">${escHtml(race.race_name_jp || race.race_name || 'Race')}</div>
        <div class="race-meta">
          ${venue    ? `<span>📍 ${escHtml(venue)}</span>` : ''}
          ${dist     ? `<span>📏 ${dist}</span>`           : ''}
          ${postTime ? `<span>🕐 ${postTime}</span>`            : ''}
          ${oddsTime ? `<span class="odds-time-tag" title="${oddsTagTitle}">${oddsTagLabel}</span>` : ''}
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
        delete raceDetailCache[race.id];
        body.innerHTML = `<div class="race-body-loading" style="color:var(--red);display:flex;align-items:center;justify-content:center;gap:10px;">
          <span>⚠️ Failed to load: ${escHtml(String(e))}</span>
          <button class="btn btn-sm" onclick="event.stopPropagation(); delete raceDetailCache[${race.id}]; const c = document.getElementById('race-card-${race.id}'); if (c) { c.classList.remove('open'); toggleRace(c, {id:${race.id}, date:'${race.date}'}); }" style="padding:3px 10px;font-size:11px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);color:var(--red);">🔄 Retry</button>
        </div>`;
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
    if (hasPrimaryBet && num) num.style.cssText = 'background:var(--accent-dim);color:#a5b4fc;border:1px solid rgba(99,102,241,0.25)';
    const hasPreds = (raceDetailCache[race.id]?.predictions?.length || 0) > 0;
    const tags = card.querySelector('.race-tags');
    if (tags) {
      const isDirt = (race.surface || '').toLowerCase() === 'dirt';
      const surfaceTag = isDirt
        ? `<span class="tag tag-dirt">Dirt</span>`
        : `<span class="tag tag-turf">Turf</span>`;
      const betTag = hasPrimaryBet
        ? `<span class="tag tag-worth-bet"><span class="flame-icon">🔥</span> VALUE BET</span>`
        : hasPreds
          ? `<span class="tag tag-predicted" style="opacity:0.65">Predicted (Pass)</span>`
          : `<span class="tag tag-nobets" style="opacity:0.5">No bet</span>`;
      tags.innerHTML = surfaceTag + betTag;
    }
  }
}

// ── Global state for betting analysis ──
let raceBudgetState = {}; // race_id -> budget integer

// ── Render race body with betting analysis, pricing, and full stats drawer ──
async function renderRaceDetail(body, detail, race) {
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

  // Model version label
  const modelVersion = predictions[0]?.model_version || '';
  if (modelVersion) {
    headerModel.textContent = modelVersion;
  }

  const raceId = race.id;
  const initialBudget = raceBudgetState[raceId] || 1000;

  body.innerHTML = `
    ${betResultHtml}
    <div class="betting-analysis-container" id="analysis-container-${raceId}">
      <div class="race-body-loading" style="padding:20px 0;">
        <div class="spinner" style="width:18px;height:18px;border-width:2px;margin:0 auto 8px;"></div>
        <span style="font-size:12px;color:var(--text-muted)">Generating betting analysis & fair odds…</span>
      </div>
    </div>

    <!-- Collapsible Full Details Drawer & Actions -->
    <div class="race-detail-actions-bar">
      <button class="stats-drawer-btn" id="drawer-btn-${raceId}" onclick="toggleStatsDrawer(${raceId})">
        <span>📋 Detailed Race Stats & Full Entries (L3F, Popularity, Finish)</span>
        <span id="drawer-icon-${raceId}">▾</span>
      </button>
      <button class="race-rescrape-btn" id="btn-rescrape-${raceId}" onclick="rescrapeSingleRace(${raceId}, '${race.date}')" title="Scrape latest netkeiba card for this race">
        <span class="btn-icon">📥</span> Rescrape Race
      </button>
    </div>
    <div class="stats-drawer-content" id="drawer-content-${raceId}">
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
      </div>
    </div>
  `;

  // Load and render betting analysis
  await updateBettingAnalysisView(raceId, initialBudget, detail);
}

// ── Toggle full details stats drawer ──
function toggleStatsDrawer(raceId) {
  const content = document.getElementById(`drawer-content-${raceId}`);
  const icon = document.getElementById(`drawer-icon-${raceId}`);
  if (!content) return;
  const isOpen = content.classList.contains('open');
  content.classList.toggle('open', !isOpen);
  if (icon) {
    icon.textContent = isOpen ? '▾' : '▴';
  }
}

// ── Track selected strategy mode per race (hybrid / pure_win / dutching) ──
const raceStrategyState = {};

// ── Update betting analysis card (supports dynamic budget and strategy mode changes) ──
async function updateBettingAnalysisView(raceId, budget, detail, mode) {
  if (budget != null) raceBudgetState[raceId] = budget;
  if (mode != null) raceStrategyState[raceId] = mode;

  const currentBudget = raceBudgetState[raceId] || 1000;
  const currentMode = raceStrategyState[raceId] || 'auto';
  const container = document.getElementById(`analysis-container-${raceId}`);
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/races/${raceId}/betting-analysis?budget=${currentBudget}&mode=${currentMode}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderBettingAnalysisHtml(container, data, raceId, currentBudget, currentMode);
  } catch (e) {
    console.warn(`Failed to fetch betting analysis for race ${raceId}:`, e);
    // Render fallback from local detail data if available
    renderLocalBettingAnalysisFallback(container, detail || raceDetailCache[raceId], raceId, currentBudget, currentMode);
  }
}

// ── Render HTML for Staking Card and Pricing Table Card ──
function renderBettingAnalysisHtml(container, data, raceId, currentBudget, currentMode = 'auto') {
  const headline = data.headline || {};
  const staking = data.staking_plan || {};
  const tickets = staking.tickets || [];
  const pricing = data.pricing_table || [];
  const allStratResults = data.all_strategies_results || {};
  const strategyResult = data.strategy_result || allStratResults[currentMode] || staking.strategy_result || {};
  const strategyMeta = data.strategy_meta || {
    name: 'Adaptive AI Router',
    icon: '🧠',
    tagline: 'AI Dynamic Per-Race Optimal Routing',
    historical_roi: '+88.5%',
    hit_rate: '38.2%',
    sharpe: 3.95,
  };

  const budgetOptions = [500, 1000, 2500, 5000, 10000];

  // Budget switchers
  const budgetChipsHtml = budgetOptions.map(b => `
    <button class="budget-chip ${b === currentBudget ? 'active' : ''}"
            onclick="updateBettingAnalysisView(${raceId}, ${b}, null, '${currentMode}')">
      ¥${b.toLocaleString()}
    </button>
  `).join('');

  // Strategy Mode Pills with live race outcome badges (if race is settled)
  const strategyModes = [
    { id: 'auto', label: 'AI Auto (Optimal)', icon: '🧠' },
    { id: 'hybrid', label: 'Balanced (Win + Exotics)', icon: '🎯' },
    { id: 'pure_win', label: 'Pure Win (Max ROI)', icon: '⚡' },
    { id: 'dutching', label: 'Dual Dutching (Low DD)', icon: '🛡️' },
  ];

  const strategyPillsHtml = `
    <div class="strategy-mode-bar">
      ${strategyModes.map(m => {
        const mRes = allStratResults[m.id];
        let resBadgeHtml = '';
        if (mRes && mRes.is_settled && mRes.staked > 0) {
          const isProfitable = mRes.profit > 0;
          const isZero = mRes.profit === 0;
          const sign = isProfitable ? '+' : '';
          const cls = isProfitable ? 'won' : isZero ? 'push' : 'lost';
          resBadgeHtml = `<span class="strat-btn-res-pill ${cls}">${isProfitable ? '✅ ' : isZero ? '⚖️ ' : '❌ '}${sign}¥${mRes.profit.toLocaleString()}</span>`;
        }
        return `
          <button class="strategy-mode-btn ${m.id === currentMode ? 'active' : ''}"
                  onclick="updateBettingAnalysisView(${raceId}, ${currentBudget}, null, '${m.id}')">
            <span>${m.icon}</span>
            <span>${m.label}</span>
            ${resBadgeHtml}
          </button>
        `;
      }).join('')}
    </div>
  `;

  // Strategy Outcome Summary Banner (shown when race results are recorded)
  let strategyOutcomeBannerHtml = '';
  if (strategyResult && strategyResult.is_settled && strategyResult.staked > 0) {
    const isWin = strategyResult.profit > 0;
    const isZero = strategyResult.profit === 0;
    const profitSign = strategyResult.profit > 0 ? '+' : '';
    const statusCls = isWin ? 'strat-outcome-win' : isZero ? 'strat-outcome-push' : 'strat-outcome-loss';
    const statusIcon = isWin ? '🏆' : isZero ? '⚖️' : '❌';
    const statusText = isWin ? 'STRATEGY WON' : isZero ? 'BREAK EVEN' : 'STRATEGY LOST';
    const roiPill = isWin
      ? `<span class="outcome-roi positive">+${strategyResult.roi_pct}% ROI</span>`
      : !isZero ? `<span class="outcome-roi negative">${strategyResult.roi_pct}% ROI</span>` : '';

    strategyOutcomeBannerHtml = `
      <div class="strategy-outcome-banner ${statusCls}">
        <div class="outcome-main-group">
          <div class="outcome-status-badge">
            <span class="outcome-icon">${statusIcon}</span>
            <span class="outcome-title">${statusText}</span>
            ${roiPill}
          </div>
          <div class="outcome-subtitle">
            ${strategyResult.tickets_won} of ${strategyResult.tickets_count} ticket${strategyResult.tickets_count === 1 ? '' : 's'} hit
          </div>
        </div>
        <div class="outcome-numbers-grid">
          <div class="outcome-stat">
            <span class="stat-lbl">STAKED</span>
            <span class="stat-val">¥${strategyResult.staked.toLocaleString()}</span>
          </div>
          <div class="outcome-stat">
            <span class="stat-lbl">PAYOUT</span>
            <span class="stat-val ${isWin ? 'text-green font-bold' : ''}">¥${strategyResult.payout.toLocaleString()}</span>
          </div>
          <div class="outcome-stat">
            <span class="stat-lbl">NET P&L</span>
            <span class="stat-val ${isWin ? 'text-green font-bold' : isZero ? 'text-neutral' : 'text-red'}">
              ${profitSign}¥${strategyResult.profit.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    `;
  }

  // Strategy Meta Insights Banner
  const strategyBannerHtml = `
    <div class="strategy-meta-banner">
      <div class="strategy-meta-left">
        <span>${strategyMeta.icon || '📊'}</span>
        <span>${escHtml(strategyMeta.tagline || strategyMeta.name)}</span>
      </div>
      <div class="strategy-meta-stats">
        <span class="strategy-meta-stat-item">Hist. ROI: <strong>${escHtml(strategyMeta.historical_roi || '+55%')}</strong></span>
        <span class="strategy-meta-stat-item">Win Rate: <strong>${escHtml(strategyMeta.hit_rate || '35%')}</strong></span>
        <span class="strategy-meta-stat-item">Sharpe: <strong>${strategyMeta.sharpe || '3.0'}</strong></span>
      </div>
    </div>
  `;

  // Staking tickets with settlement & runner finish details
  const ticketsHtml = tickets.map(t => {
    const typeCls = `ticket-type-${t.type || '単勝'}`;
    const probPct = Math.round((t.prob || 0) * 100);
    const isLivePool = t.odds_source === 'live_pool';
    const sourceBadge = isLivePool
      ? `<span class="odds-source-badge live" title="Live JRA betting pool dividend">LIVE POOL</span>`
      : `<span class="odds-source-badge est" title="Theoretical estimate synthesized from Win odds">EST</span>`;
    
    let mktInfo = '';
    if (t.market_odds) {
      const estHint = (isLivePool && t.estimated_market_odds && t.estimated_market_odds !== t.market_odds)
        ? ` <span class="est-hint" title="Theoretical Harville Model baseline">(Est: ${t.estimated_market_odds}x)</span>`
        : '';
      mktInfo = ` · ${isLivePool ? 'Mkt' : 'Est. Mkt'}: <strong>${t.market_odds}x</strong> ${sourceBadge}${estHint}`;
    }
    
    const evInfo = t.ev != null ? ` · EV: <strong style="color:${t.ev > 0 ? '#22c55e' : '#f87171'}">${t.ev > 0 ? '+' : ''}${(t.ev * 100).toFixed(0)}%</strong>` : '';
    
    // Settlement Information
    let cardStateClass = '';
    let resultTagHtml = '';
    let runnerFinishBadgesHtml = '';
    let stakePayoutHtml = '';

    if (t.is_settled) {
      if (t.won) {
        cardStateClass = 'ticket-won';
        const netProfit = t.profit != null ? t.profit : (t.payout - t.stake);
        const netSign = netProfit > 0 ? '+' : '';
        resultTagHtml = `<span class="ticket-result-pill won">✅ WON · +¥${(t.payout || 0).toLocaleString()} (Net ${netSign}¥${netProfit.toLocaleString()})</span>`;
      } else {
        cardStateClass = 'ticket-lost';
        resultTagHtml = `<span class="ticket-result-pill lost">❌ LOST · ¥0 (Net -¥${(t.stake || 0).toLocaleString()})</span>`;
      }

      // Runner finish indicators
      if (t.runner_finishes && t.runner_finishes.length > 0) {
        runnerFinishBadgesHtml = `
          <div class="ticket-runner-finishes">
            ${t.runner_finishes.map(rf => {
              const pos = rf.finish_pos;
              let posHtml;
              if (pos === 1) posHtml = '<span class="finish-pill finish-1">🥇 1st</span>';
              else if (pos === 2) posHtml = '<span class="finish-pill finish-2">🥈 2nd</span>';
              else if (pos === 3) posHtml = '<span class="finish-pill finish-3">🥉 3rd</span>';
              else if (pos != null) posHtml = `<span class="finish-pill finish-other">${pos}th</span>`;
              else posHtml = '<span class="finish-pill finish-dnf">—</span>';
              return `<span class="ticket-runner-tag">#${rf.post_position} ${escHtml(rf.horse_name)} ${posHtml}</span>`;
            }).join(' ')}
          </div>
        `;
      }

      // Stake & Payout box
      if (t.won) {
        stakePayoutHtml = `
          <div class="ticket-stake-payout won">
            <div class="ticket-stake-sub">Staked ¥${(t.stake || 0).toLocaleString()}</div>
            <div class="ticket-payout-val">+¥${(t.payout || 0).toLocaleString()}</div>
          </div>
        `;
      } else {
        stakePayoutHtml = `
          <div class="ticket-stake-payout lost">
            <div class="ticket-stake-sub">Staked ¥${(t.stake || 0).toLocaleString()}</div>
            <div class="ticket-payout-val zero">¥0</div>
          </div>
        `;
      }
    } else {
      stakePayoutHtml = `<div class="ticket-stake">¥${(t.stake || 0).toLocaleString()}</div>`;
    }

    return `
      <div class="ticket-card ${cardStateClass}">
        <div class="ticket-left">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span class="ticket-type-badge ${typeCls}">${escHtml(t.type)}</span>
            <span class="ticket-display">${escHtml(t.label.split('—')[1]?.trim() || t.label)}</span>
            ${resultTagHtml}
          </div>
          <div class="ticket-stats">Win chance: <strong>${probPct}%</strong> · Fair: <strong>${t.fair_odds}x</strong>${mktInfo}${evInfo}</div>
          ${runnerFinishBadgesHtml}
        </div>
        ${stakePayoutHtml}
      </div>
    `;
  }).join('');

  // Simple Bet settlement info
  let simpleBetStatusHtml = '';
  const sb = staking.simple_bet;
  if (sb && sb.is_settled) {
    if (sb.won) {
      const net = sb.profit != null ? sb.profit : (sb.payout - sb.stake);
      simpleBetStatusHtml = ` <strong style="color:#4ade80;margin-left:6px;">→ ✅ WON +¥${(sb.payout||0).toLocaleString()} (Net +¥${net.toLocaleString()} · 🥇 1st)</strong>`;
    } else {
      const fpos = sb.runner_finishes?.[0]?.finish_pos;
      const fposStr = fpos ? `Finished ${ordinal(fpos)}` : 'Did not place 1st';
      simpleBetStatusHtml = ` <strong style="color:#f87171;margin-left:6px;">→ ❌ LOST (${fposStr})</strong>`;
    }
  }

  // Pricing Table rows with Finish Positions & Winner Highlights
  const pricingRowsHtml = pricing.map(h => {
    const verdictCls = getVerdictBadgeClass(h.verdict);
    const probPct = h.win_prob_pct || Math.round((h.win_prob || 0) * 100);
    const ppHtml = (h.post_position && h.post_position > 0)
      ? `<span class="pos-num">${h.post_position}</span>`
      : `<span class="pos-num pos-pending" title="Gate pending draw">—</span>`;
    const mktOddsHtml = (h.market_odds != null && h.market_odds > 1.0)
      ? `${h.market_odds}x`
      : `<span style="color:var(--text-dim);font-weight:normal">Pending</span>`;

    const finishPos = h.finish_pos;
    let finishBadgeHtml = '';
    if (finishPos === 1) finishBadgeHtml = '<span class="finish-badge finish-1">🥇 1st</span>';
    else if (finishPos === 2) finishBadgeHtml = '<span class="finish-badge finish-2">🥈 2nd</span>';
    else if (finishPos === 3) finishBadgeHtml = '<span class="finish-badge finish-3">🥉 3rd</span>';
    else if (finishPos != null) finishBadgeHtml = `<span class="finish-badge finish-other">${finishPos}th</span>`;
    else finishBadgeHtml = '<span class="finish-badge finish-dnf" style="opacity:0.4">—</span>';

    const isRowWinner = finishPos === 1;
    const rowCls = isRowWinner ? 'pricing-row-winner is-winner' : '';

    return `
      <tr class="${rowCls}">
        <td>${ppHtml}</td>
        <td>
          <div class="horse-name">${escHtml(h.horse_name)}</div>
        </td>
        <td>
          <div class="prob-bar-wrap">
            <div class="prob-bar"><div class="prob-bar-fill" style="width:${Math.min(100, probPct)}%"></div></div>
            <span class="prob-text">${probPct}%</span>
          </div>
        </td>
        <td style="font-weight:600;color:#93c5fd">${h.fair_odds}x</td>
        <td style="font-weight:600">${mktOddsHtml}</td>
        <td>${finishBadgeHtml}</td>
        <td><span class="verdict-badge ${verdictCls}">${escHtml(h.verdict)}</span></td>
      </tr>
    `;
  }).join('');

  let ticketsBodyHtml = '';
  const hasMktOdds = pricing.some(h => h.market_odds != null && h.market_odds > 1.0);
  if (!hasMktOdds) {
    ticketsBodyHtml = `
      <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:8px;padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
        <span style="font-size:22px;">⏳</span>
        <div>
          <div style="font-size:14px;font-weight:700;color:#a5b4fc;margin-bottom:2px;">Market Odds Pending (Draw & Betting Opens Friday)</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.4;">Official JRA betting pools open Friday evening. Model win probabilities and fair decimal odds are calculated in the table below. Ticket staking and Kelly sizing will unlock once market odds are published.</div>
        </div>
      </div>
    `;
  } else if (tickets.length === 0) {
    ticketsBodyHtml = `
      <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:8px;padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
        <span style="font-size:22px;">🛑</span>
        <div>
          <div style="font-size:14px;font-weight:700;color:#fca5a5;margin-bottom:2px;">Pass — No Mathematical Edge</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.4;">Every horse and exotic combination in this race is fairly priced or underpriced by the market. Bankroll is preserved for positive-EV opportunities.</div>
        </div>
      </div>
    `;
  } else {
    ticketsBodyHtml = `
      <div class="ticket-grid">
        ${ticketsHtml}
      </div>
      <div class="simple-bet-box">
        <span class="simple-bet-icon">💡</span>
        <span>${escHtml(staking.simple_bet_label || 'Simple bet option available')}${simpleBetStatusHtml}</span>
      </div>
    `;
  }

  const oddsTime = formatOddsTime(data.odds_as_of || data.odds_updated_at || raceDetailCache[raceId]?.race?.odds_as_of || raceDetailCache[raceId]?.race?.odds_updated_at);
  const oddsAsOfHtml = oddsTime
    ? `<span class="odds-as-of-badge" title="Market odds captured as of ${escHtml(oddsTime)}">⏱ Odds as of <strong>${escHtml(oddsTime)}</strong></span>`
    : '';

  container.innerHTML = `
    <!-- 1. Executive Staking Card -->
    <div class="staking-card">
      <div class="staking-header">
        <div class="staking-headline-group">
          <span class="staking-title">🎯 Executive Betting Portfolio</span>
          <div class="staking-recommendation">
            <span>👑 ${escHtml(headline.recommended_horse || 'Top Pick')}</span>
            <span style="font-size:13px;font-weight:600;color:#a5b4fc;background:rgba(99,102,241,0.15);padding:2px 8px;border-radius:12px;border:1px solid rgba(99,102,241,0.3);">${escHtml(headline.action || 'to win')}</span>
          </div>
        </div>
        <div class="staking-budget-control">
          <span class="budget-label">Budget:</span>
          <div class="budget-chips-wrap">
            ${budgetChipsHtml}
            <div class="custom-budget-container">
              <span class="custom-budget-curr">¥</span>
              <input type="number"
                     class="custom-budget-input"
                     id="custom-budget-input-${raceId}"
                     min="100"
                     max="1000000"
                     step="500"
                     placeholder="Custom"
                     value="${currentBudget}"
                     onchange="updateBettingAnalysisView(${raceId}, Math.max(100, parseInt(this.value)||1000), null, '${currentMode}')"
                     onkeydown="if(event.key==='Enter'){updateBettingAnalysisView(${raceId}, Math.max(100, parseInt(this.value)||1000), null, '${currentMode}');}"
                     title="Type custom bankroll amount in yen">
            </div>
          </div>
        </div>
      </div>

      ${strategyPillsHtml}
      ${strategyOutcomeBannerHtml}
      ${strategyBannerHtml}
      ${ticketsBodyHtml}
    </div>

    <!-- 2. My Pricing Card -->
    <div class="pricing-card">
      <div class="pricing-header">
        <div class="pricing-title">
          <span>📊 Fair Odds Pricing & Analytical Verdicts</span>
        </div>
        ${oddsAsOfHtml}
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Horse</th>
              <th>Win Chance</th>
              <th>Fair Odds</th>
              <th>Market Odds${oddsTime ? ` <span class="odds-col-time">(${escHtml(oddsTime)})</span>` : ''}</th>
              <th>Finish</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            ${pricingRowsHtml}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ── Verdict badge CSS class mapper ──
function getVerdictBadgeClass(verdict) {
  if (!verdict) return 'verdict-fair';
  const v = verdict.toLowerCase();
  if (v.includes('top pick') || v.includes('model top pick')) return 'verdict-best-value';
  if (v.includes('best value') || v.includes('moderate value')) return 'verdict-best-value';
  if (v.includes('underpriced')) return 'verdict-underpriced';
  if (v.includes('secondary value') || v.includes('slight value') || v.includes('contender')) return 'verdict-value';
  if (v.includes('longshot')) return 'verdict-longshot';
  if (v.includes('clearly too short')) return 'verdict-too-short';
  if (v.includes('slightly short') || v.includes('short')) return 'verdict-short';
  return 'verdict-fair';
}

// ── Local Fallback for offline/cached states ──
function renderLocalBettingAnalysisFallback(container, detail, raceId, budget, mode = 'auto') {
  const entries = detail?.entries || [];
  const predictions = detail?.predictions || [];
  const predMap = {};
  predictions.forEach(p => { predMap[p.entry_id] = p; });

  const priced = entries.map(e => {
    const pred = predMap[e.id] || {};
    const prob = Math.max(0.0001, parseFloat(pred.win_prob || 0.05));
    const mkt = parseFloat(e.odds_win || 1.0);
    const fair = Math.round((1.0 / prob) * 10) / 10;
    const ev = (prob * mkt) - 1.0;
    let verdict;
    if (ev >= 0.10) verdict = "Best value";
    else if (ev > 0.05) verdict = "Moderate value";
    else if (ev > 0.0) verdict = "Slight value";
    else if (ev >= -0.05) verdict = "Roughly fair";
    else if (ev >= -0.20) verdict = "Slightly short";
    else verdict = "Clearly too short";
    return {
      post_position: e.post_position,
      horse_name: e.horse_name_jp || e.horse_name || `Horse ${e.post_position}`,
      win_prob_pct: Math.round(prob * 100),
      win_prob: prob,
      fair_odds: fair,
      market_odds: mkt,
      ev: ev,
      verdict: verdict,
      finish_pos: e.finish_pos,
    };
  });

  priced.sort((a, b) => b.win_prob - a.win_prob);
  const anchor = priced.find(p => p.ev >= 0.05) || priced[0] || { post_position: 1, horse_name: "Top Runner", win_prob: 0.25, fair_odds: 4.0, finish_pos: null };

  const isSettled = entries.some(e => e.finish_pos != null);
  const anchorWon = anchor.finish_pos === 1;
  const anchorPayout = anchorWon ? Math.round(budget * (anchor.market_odds || anchor.fair_odds)) : 0;
  const anchorProfit = anchorPayout - budget;

  const tickets = [
    {
      type: "単勝",
      type_en: "win",
      selection: [anchor.post_position],
      selection_display: `No. ${anchor.post_position} 単勝`,
      label: `¥${budget.toLocaleString()} — No. ${anchor.post_position} 単勝`,
      stake: budget,
      prob: anchor.win_prob,
      fair_odds: anchor.fair_odds,
      market_odds: anchor.market_odds,
      ev: anchor.ev,
      is_settled: isSettled,
      won: anchorWon,
      payout: anchorPayout,
      profit: anchorProfit,
      roi_pct: round1(anchorProfit / budget * 100),
      runner_finishes: [
        { post_position: anchor.post_position, finish_pos: anchor.finish_pos, horse_name: anchor.horse_name }
      ],
    }
  ];

  const stratResult = {
    is_settled: isSettled,
    staked: budget,
    payout: anchorPayout,
    profit: anchorProfit,
    roi_pct: round1(anchorProfit / budget * 100),
    status: anchorWon ? "won" : "lost",
    tickets_won: anchorWon ? 1 : 0,
    tickets_count: 1,
  };

  const data = {
    headline: {
      recommended_horse: `No. ${anchor.post_position} ${anchor.horse_name}`,
      post_position: anchor.post_position,
      action: "to win",
      summary: `My bet: No. ${anchor.post_position} ${anchor.horse_name} to win`,
    },
    strategy_result: stratResult,
    all_strategies_results: {
      auto: stratResult,
      pure_win: stratResult,
      hybrid: stratResult,
      dutching: stratResult,
    },
    staking_plan: {
      budget: budget,
      tickets: tickets,
      simple_bet: tickets[0],
      simple_bet_label: `If you only want one uncomplicated bet: ¥${budget.toLocaleString()} on No. ${anchor.post_position} 単勝.`,
      strategy_result: stratResult,
    },
    pricing_table: priced,
    odds_as_of: detail?.race?.odds_as_of || detail?.race?.odds_updated_at || detail?.predictions?.[0]?.created_at,
  };

  renderBettingAnalysisHtml(container, data, raceId, budget, mode);
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
  const pp     = (e.post_position && e.post_position > 0) ? e.post_position : '—';
  const ppCls  = pp === '—' ? 'pos-pending' : '';
  const popStr = (e.popularity != null && e.popularity > 0) ? `Pop. ${e.popularity}` : 'Pop. —';

  // 📌 = the one actual bet; ◆ VALUE = clears threshold but not top pick
  const betHtml = isPrimaryBet
    ? `<span class="bet-marker">📌 BET</span>`
    : isValueOnly
      ? `<span class="bet-marker value-only" title="Clears EV threshold but not the top pick">◆ VALUE</span>`
      : '';

  return `
    <tr class="${cls}">
      <td><span class="pos-num ${ppCls}" title="${pp === '—' ? 'Gate pending draw' : ''}">${pp}</span></td>
      <td>
        <div class="horse-name">${escHtml(e.horse_name_jp || e.horse_name || '—')}</div>
        <div class="horse-meta">${popStr}</div>
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
function statCard(label, value, sub, cls, extraAttr = '') {
  if (value === undefined || value === null || value === '') return '';
  return `
    <div class="stat-card ${cls || ''}" ${extraAttr}>
      <div class="stat-label">${label}</div>
      <div class="stat-value ${cls || ''}">${value}</div>
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

function formatOddsTime(ts) {
  if (!ts) return '';
  const s = String(ts).trim();
  if (!s || s === 'None' || s === 'null') return '';

  // Format "YYYY-MM-DD HH:MM:SS" (Netkeiba JST timestamp)
  if (s.includes(' ') && s.includes(':')) {
    const timePart = s.split(' ')[1];
    const parts = timePart.split(':');
    return `${parts[0]}:${parts[1]}`;
  }
  // ISO format "YYYY-MM-DDTHH:MM:SS..." (UTC created_at)
  if (s.includes('T')) {
    const d = new Date(s.endsWith('Z') ? s : s + 'Z');
    if (!isNaN(d.getTime())) {
      // JST is UTC+9
      const jstHours = (d.getUTCHours() + 9) % 24;
      const jstMins = d.getUTCMinutes();
      return `${String(jstHours).padStart(2, '0')}:${String(jstMins).padStart(2, '0')}`;
    }
  }
  const match = s.match(/(\d{1,2}):(\d{2})/);
  if (match) return `${match[1].padStart(2, '0')}:${match[2]}`;
  return s;
}

// ── Start ──
init();
