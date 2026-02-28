// UmaEdge Results App
(async function () {
    const resp = await fetch('data.json');
    const DATA = await resp.json();

    let activeVenue = 'all';
    let activeFilter = 'all';
    let sortCol = 'ev';
    let sortDir = 'desc';

    // ── Summary Cards ──
    function renderSummary() {
        const s = DATA.summary;
        const cards = [
            { icon: '🏁', value: DATA.total_races, label: 'Races' },
            { icon: '🐴', value: DATA.total_entries, label: 'Entries' },
            { icon: '💎', value: s.total_bets, label: 'Value Bets' },
            { icon: '🏆', value: s.winners, label: 'Winners Hit' },
            { icon: '🎯', value: s.strike_rate + '%', label: 'Strike Rate' },
            { icon: '💰', value: '¥' + s.total_stake.toLocaleString(), label: 'Total Stake' },
            { icon: '📈', value: '¥' + s.total_returns.toLocaleString(), label: 'Returns', cls: s.total_returns > s.total_stake ? 'positive' : 'negative' },
            { icon: '📊', value: (s.roi > 0 ? '+' : '') + s.roi + '%', label: 'ROI', cls: s.roi > 0 ? 'positive' : 'negative' },
        ];
        document.getElementById('summaryGrid').innerHTML = cards.map(c => `
            <div class="summary-card">
                <div class="summary-icon">${c.icon}</div>
                <div class="summary-value ${c.cls || ''}">${c.value}</div>
                <div class="summary-label">${c.label}</div>
            </div>
        `).join('');
    }

    // ── Venue Tabs ──
    function renderVenueTabs() {
        const venues = ['all', ...Object.keys(DATA.venues)];
        const labels = { all: 'All Venues', Hanshin: '🟣 阪神', Kokura: '🟢 小倉', Nakayama: '🔵 中山' };
        document.getElementById('venueTabs').innerHTML = venues.map(v => `
            <button class="venue-tab ${v === activeVenue ? 'active' : ''}" data-venue="${v}">
                ${labels[v] || v}
            </button>
        `).join('');

        document.querySelectorAll('.venue-tab').forEach(btn => {
            btn.addEventListener('click', () => {
                activeVenue = btn.dataset.venue;
                renderVenueTabs();
                renderRaces();
            });
        });
    }

    // ── Filter Buttons ──
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeFilter = btn.dataset.filter;
            renderRaces();
        });
    });

    // ── Race Cards ──
    function renderRaces() {
        const grid = document.getElementById('racesGrid');
        let html = '';

        const venues = activeVenue === 'all' ? Object.keys(DATA.venues) : [activeVenue];

        for (const venue of venues) {
            const races = DATA.venues[venue] || [];
            for (const race of races) {
                const hasValueBets = race.entries.some(e => e.is_value_bet);
                const topPick = race.entries[0];

                if (activeFilter === 'value' && !hasValueBets) continue;
                if (activeFilter === 'top' && (!topPick || topPick.prob_combined < 30)) continue;

                const valueBetCount = race.entries.filter(e => e.is_value_bet).length;
                const winnerEntry = race.entries.find(e => e.is_winner);

                html += `
                <div class="race-card" id="race-${venue}-${race.race_number}">
                    <div class="race-header" onclick="toggleRace(this)">
                        <div class="race-title-group">
                            <div class="race-number">R${race.race_number}</div>
                            <div>
                                <div class="race-name">${race.race_name}</div>
                                <div class="race-meta">
                                    <span class="race-badge">${venue}</span>
                                    <span class="race-badge ${race.surface}">${race.surface === 'turf' ? '芝' : 'ダ'}${race.distance}m</span>
                                    <span class="race-badge">${race.going || ''}</span>
                                    <span class="race-badge">${race.field_size}頭</span>
                                    ${valueBetCount ? `<span class="race-badge" style="background:var(--accent-glow);color:var(--accent-light)">💎 ${valueBetCount}</span>` : ''}
                                </div>
                            </div>
                        </div>
                        <div class="race-stats">
                            ${winnerEntry ? `<div class="race-stat"><div class="race-stat-value" style="color:var(--gold)">🏆 ${winnerEntry.horse_name}</div><div class="race-stat-label">${winnerEntry.odds ? winnerEntry.odds.toFixed(1) + 'x' : ''}</div></div>` : ''}
                            <div class="race-stat">
                                <div class="race-stat-value">${topPick ? topPick.prob_combined + '%' : '--'}</div>
                                <div class="race-stat-label">Top Prob</div>
                            </div>
                            <span class="expand-icon">▼</span>
                        </div>
                    </div>
                    <div class="race-body">
                        <table class="entry-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>PP</th>
                                    <th>Horse</th>
                                    <th>Odds</th>
                                    <th>Pop</th>
                                    <th>Model %</th>
                                    <th></th>
                                    <th>Fund %</th>
                                    <th>Mkt %</th>
                                    <th>EV</th>
                                    <th>Result</th>
                                    <th>Time</th>
                                    <th>3F</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${race.entries.map(e => renderEntryRow(e)).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>`;
            }
        }
        grid.innerHTML = html || '<p style="text-align:center;color:var(--text-muted);padding:40px;">No races match the current filter.</p>';
    }

    function renderEntryRow(e) {
        const evClass = e.ev && e.ev > 0 ? 'ev-positive' : 'ev-negative';
        const rowClass = `${e.is_value_bet ? 'value-bet' : ''} ${e.is_winner ? 'winner' : ''}`;
        const finClass = e.finish_pos === 1 ? 'finish-1' : e.finish_pos === 2 ? 'finish-2' : e.finish_pos === 3 ? 'finish-3' : 'finish-other';

        return `<tr class="${rowClass}">
            <td>${e.finish_pos ? `<span class="finish-badge ${finClass}">${e.finish_pos}</span>` : '<span style="color:var(--text-muted)">—</span>'}</td>
            <td style="color:var(--text-muted)">${e.post_position || '—'}</td>
            <td class="horse-name">${e.horse_name}${e.is_winner ? '<span class="winner-badge">👑</span>' : ''}</td>
            <td>${e.odds ? e.odds.toFixed(1) : '—'}</td>
            <td style="color:var(--text-muted)">${e.popularity || '—'}</td>
            <td><strong>${e.prob_combined}%</strong></td>
            <td><div class="prob-bar"><div class="prob-bar-fill" style="width:${Math.min(e.prob_combined * 1.5, 100)}%"></div></div></td>
            <td style="color:var(--text-muted)">${e.prob_fund}%</td>
            <td style="color:var(--text-muted)">${e.prob_mkt}%</td>
            <td>${e.ev != null ? `<span class="ev-badge ${evClass}">${e.ev > 0 ? '+' : ''}${e.ev}%</span>` : '—'}</td>
            <td>${e.finish_pos ? `<span class="finish-badge ${finClass}">${e.finish_pos}</span>` : '—'}</td>
            <td style="color:var(--text-secondary)">${e.time_secs || '—'}</td>
            <td style="color:var(--text-secondary)">${e.last_3f || '—'}</td>
        </tr>`;
    }

    // ── Value Bets Table ──
    function renderValueBets() {
        const tbody = document.getElementById('valueTableBody');
        const bets = [...DATA.value_bets];

        bets.sort((a, b) => {
            let va = a[sortCol], vb = b[sortCol];
            if (va == null) va = -Infinity;
            if (vb == null) vb = -Infinity;
            return sortDir === 'desc' ? vb - va : va - vb;
        });

        document.getElementById('valueBetCount').textContent = bets.length;

        tbody.innerHTML = bets.map(b => {
            const evClass = b.ev > 20 ? 'ev-positive' : 'ev-positive';
            const finClass = b.finish_pos === 1 ? 'finish-1' : b.finish_pos === 2 ? 'finish-2' : b.finish_pos === 3 ? 'finish-3' : 'finish-other';
            const rowClass = b.is_winner ? 'winner-row' : '';
            return `<tr class="${rowClass}">
                <td>${b.venue}</td>
                <td>R${b.race_number}</td>
                <td class="horse-name">${b.horse_name}${b.is_winner ? ' 👑' : ''}</td>
                <td>${b.post_position || '—'}</td>
                <td>${b.odds ? b.odds.toFixed(1) + 'x' : '—'}</td>
                <td>${b.prob_combined}%</td>
                <td><span class="ev-badge ev-positive">+${b.ev}%</span></td>
                <td>${b.finish_pos ? `<span class="finish-badge ${finClass}">${b.finish_pos}</span>` : '—'}</td>
            </tr>`;
        }).join('');
    }

    // Sort value table
    document.querySelectorAll('.value-table th').forEach((th, i) => {
        const cols = ['venue', 'race_number', 'horse_name', 'post_position', 'odds', 'prob_combined', 'ev', 'finish_pos'];
        th.addEventListener('click', () => {
            const col = cols[i];
            if (sortCol === col) { sortDir = sortDir === 'desc' ? 'asc' : 'desc'; }
            else { sortCol = col; sortDir = 'desc'; }
            document.querySelectorAll('.value-table th').forEach(t => t.classList.remove('sorted-asc', 'sorted-desc'));
            th.classList.add(sortDir === 'desc' ? 'sorted-desc' : 'sorted-asc');
            renderValueBets();
        });
    });

    // Toggle race card expansion
    window.toggleRace = function (header) {
        header.closest('.race-card').classList.toggle('expanded');
    };

    // Initialize
    renderSummary();
    renderVenueTabs();
    renderRaces();
    renderValueBets();

    // Auto-expand first race
    const firstCard = document.querySelector('.race-card');
    if (firstCard) firstCard.classList.add('expanded');
})();
