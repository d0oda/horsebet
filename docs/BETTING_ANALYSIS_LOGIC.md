# UmaEdge: Betting Analysis, Pricing & Multi-Ticket Staking Engine

This specification details the mathematical models, algorithmic logic, and software architecture to generate actionable betting recommendations, fair odds valuations, verdict classifications, and multi-ticket portfolio staking for Japanese horse racing (JRA).

---

## 1. Overview & Objectives

Transform raw horse win probabilities and live market odds into a structured betting advisory report:
1. **Pricing & Valuation**: Calculate fair decimal odds ($1/p$) and classify every horse into an actionable verdict category.
2. **Multi-Finishing Probability Modeling**: Model joint probabilities for 1st, 2nd, and 3rd place finishes using the Harville / Plackett-Luce formulation with Henery discount corrections.
3. **All JRA Ticket Types**: Support 単勝 (Win), 複勝 (Place), ワイド (Wide / Place Quinella), 馬連 (Quinella), 馬単 (Exacta), 三連複 (Trio), and 三連単 (Trifecta).
4. **Budget-Constrained Multi-Ticket Staking**: Given any budget $B$ (default: ¥1,000, discrete in ¥100 units), construct an optimal portfolio balancing a core value Win bet, Wide hedges, and Trio/Trifecta combinations, alongside an uncomplicated single-bet alternative.

---

## 2. Mathematical Formulations

### 2.1. Fair Odds & Expected Value (Win)

For each horse $i \in \{1, \dots, N\}$ in a race of $N$ runners:
* Let $p_i$ be the model's calibrated win probability, where $\sum_{i=1}^N p_i = 1.0$.
* Let $O_i$ be the current live/screenshot decimal market odds.

$$\text{Fair Odds}_i = \frac{1}{\max(p_i, 10^{-4})}$$

$$\text{EV}_i = (p_i \times O_i) - 1.0 = \frac{O_i}{\text{Fair Odds}_i} - 1.0$$

$$\text{Price Ratio}_i = \frac{O_i}{\text{Fair Odds}_i} = p_i \times O_i$$

---

### 2.2. Multi-Finishing Order Probabilities (Exotic Bets)

To price exotic tickets (Place, Wide, Quinella, Exacta, Trio, Trifecta), we require the joint probability distributions of runners finishing in positions 1, 2, and 3.

#### A. Harville / Plackett-Luce Model
Under the Luce Choice Axiom, the probability that horse $j$ finishes 2nd given horse $i$ finishes 1st is proportional to their initial win probabilities:

$$P(1\text{st}=i, 2\text{nd}=j) = p_i \cdot \frac{p_j}{1 - p_i} \quad (i \ne j)$$

$$P(1\text{st}=i, 2\text{nd}=j, 3\text{rd}=k) = p_i \cdot \frac{p_j}{1 - p_i} \cdot \frac{p_k}{1 - p_i - p_j} \quad (i \ne j \ne k)$$

#### B. Henery / Discounted Power Adjustment
Standard Harville tends to slightly overestimate the place probability of extreme longshots. To adjust for this, we support a discounted Harville parameter $\gamma \in [0.80, 1.00]$:

$$P(2\text{nd}=j \mid 1\text{st}=i) = \frac{p_j^\gamma}{\sum_{m \ne i} p_m^\gamma}$$

$$P(3\text{rd}=k \mid 1\text{st}=i, 2\text{nd}=j) = \frac{p_k^\gamma}{\sum_{m \ne i, j} p_m^\gamma}$$

---

### 2.3. Probability by Ticket Type

| JRA Ticket Type | Japanese Name | Winning Condition | Joint Probability Formula $P(\text{Ticket})$ |
| :--- | :--- | :--- | :--- |
| **Win** | 単勝 (Tansho) | Horse $i$ finishes 1st | $P(i) = p_i$ |
| **Place** | 複勝 (Fukusho) | Horse $i$ finishes in Top 3 (or Top 2 if $N \le 7$) | $P(i \in \text{Top 3}) = p_i + \sum_{j \ne i} P(j, i) + \sum_{j \ne i}\sum_{k \ne i,j} P(j, k, i)$ |
| **Quinella** | 馬連 (Umaren) | $\{i, j\}$ finish 1st and 2nd (any order) | $P(\{i, j\}) = P(i, j) + P(j, i)$ |
| **Exacta** | 馬単 (Umatan) | Horse $i$ 1st, Horse $j$ 2nd (exact order) | $P(i \to j) = P(i, j)$ |
| **Wide** | ワイド (Wide) | Both $\{i, j\}$ finish in Top 3 | $P(\{i, j\} \subset \text{Top 3}) = \sum_{k \notin \{i,j\}} [P(i, j, k) + P(i, k, j) + P(j, i, k) + P(j, k, i) + P(k, i, j) + P(k, j, i)]$ |
| **Trio** | 三連複 (Sanrenpuku) | $\{i, j, k\}$ finish in Top 3 (any order) | $P(\{i, j, k\}) = \sum_{\sigma \in \text{Perm}(i,j,k)} P(\sigma_1, \sigma_2, \sigma_3)$ |
| **Trifecta** | 三連単 (Sanrentan) | $i$ 1st, $j$ 2nd, $k$ 3rd (exact order) | $P(i \to j \to k) = P(i, j, k)$ |

---

## 3. Verdict Classification Engine

For each horse $i$, determine its analytical role and verdict based on its win probability $p_i$, market odds $O_i$, and Expected Value $\text{EV}_i$.

### 3.1. Role Flags
* **Nominal Favorite ($F$)**: $F = \arg\max_i (p_i)$, provided $\max_i(p_i) > 0.06$. In wide-open fields (e.g. 18-horse handicaps) no runner may exceed 12%; the 0.06 floor ensures the leading runner is still identified rather than leaving the favorite role unfilled.
* **Top Value Runner ($V$)**: $V = \arg\max_i (\text{EV}_i \mid p_i \ge 0.04 \text{ and } \text{EV}_i > \text{EV\_TOP\_VALUE}=0.05)$. If no horse clears the EV threshold, $V$ is undefined and no is_top_value flag is set.
* **Market Favorite ($M$)**: $M = \arg\min_i (O_i)$ among horses with quoted market odds.

### 3.2. Verdict Decision Rules

```
                      ┌──────────────────────────────┐
                      │    Horse i Evaluation        │
                      └──────────────┬───────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        Is Top Value Pick (i = V)             Other Runners (i ≠ V)
                 │                                       │
     ┌───────────┴──────────────┐          ┌────────────┴────────────┐
     ▼                          ▼          ▼                         ▼
Is Fav (i=F)            Is Not Fav    Is Fav (i=F)             Not Fav
EV ≥ 0.10:              EV ≥ 0.10:    EV < 0:                  EV ≥ 0.12 & p≥5%: "Secondary value"
"Top pick &             "Best value"  "Most likely,            EV ≥ 0.10: "Secondary value"
clear value"            0.05<EV<0.10: underpriced"             0 < EV < 0.10: "Slight value"
                        "Moderate     EV ≥ 0 (not top):        p<5% & EV≥25% & O≥20: "Longshot overlay"
                        value"         → generic rules →       -0.05 ≤ EV ≤ 0: "Roughly fair"
                                                               -0.20 ≤ EV < -0.05: "Slightly short"
                                                               EV < -0.20: "Clearly too short"
```

| Rule Condition | Verdict String | Tag Badge CSS |
| :--- | :--- | :--- |
| $i = V$ and $i = F$ and $\text{EV}_i \ge 0.10$ | `Top pick & clear value` | `verdict-best-value` |
| $i = V$ and $\text{EV}_i \ge 0.10$ | `Best value` | `verdict-best-value` |
| $i = V$ and $0.05 < \text{EV}_i < 0.10$ | `Moderate value` | `verdict-moderate` |
| $i = F$ and $\text{EV}_i < 0.0$ | `Most likely, but underpriced` | `verdict-underpriced` |
| $i \ne V$ and $\text{EV}_i \ge 0.12$ and $p_i \ge 0.05$ | `Secondary value` | `verdict-value` |
| $p_i < 0.05$ and $\text{EV}_i \ge 0.25$ and $O_i \ge 20.0$ | `Longshot overlay` | `verdict-longshot` |
| $0 < \text{EV}_i < 0.10$ (none of the above) | `Slight value` | `verdict-slight` |
| $-0.05 \le \text{EV}_i \le 0.0$ | `Roughly fair` | `verdict-fair` |
| $-0.20 \le \text{EV}_i < -0.05$ | `Slightly short` | `verdict-short` |
| $\text{EV}_i < -0.20$ | `Clearly too short` | `verdict-too-short` |

**Pre-market verdicts** (when market odds are not yet available):

| Rank in field | Verdict String |
| :--- | :--- |
| Rank 1 (model favourite) | `Model Top Pick (Fair: Xx)` |
| Rank 2–3 | `Contender (Fair: Xx)` |
| Rank 4–6 | `Mid-tier (Fair: Xx)` |
| Rank 7+ | `Longshot (Fair: Xx)` |

---

## 4. Multi-Ticket Staking Engine

### 4.1. Budget Allocation Strategy (Standard ¥1,000 Portfolio)

The portfolio is anchored around:
* **Key Horse ($V$)**: The primary value selection.
* **Chalk Partner ($F$)**: The highest probability contender.
* **Secondary Contenders ($C_1, C_2$)**: High-probability or positive-EV contenders.

#### Allocation Rules for Standard ¥1,000:
1. **Core Win ($V$ 単勝)**: 50% to 60% of total budget ($\approx ¥600$).
2. **Key-Chalk Wide ($V - F$ ワイド)**: 20% of total budget ($\approx ¥200$).
3. **Key-Secondary Wide ($V - C_1$ ワイド)**: 10% of total budget ($\approx ¥100$).
4. **Trio Combination ($V - F - C_1$ 三連複)**: 10% of total budget ($\approx ¥100$).
5. **Simple Bet Alternative**: $100\%$ on $V$ 単勝 (¥1,000).

### 4.2. Generalized Budget Scaling & ¥100 Integer Discretization

For any arbitrary budget $B \ge 500$:
1. Compute raw floating-point allocations: $a_k = B \times w_k$.
2. Floor each allocation to the nearest ¥100 unit: $s_k = \lfloor a_k / 100 \rfloor \times 100$.
3. Compute remaining undistributed balance: $R = B - \sum s_k$.
4. Distribute remainder in ¥100 increments to tickets sorted by their Expected Value / Weight descending until $\sum s_k = B$.

---

## 5. Implementation Code Structure

### 5.1. Core Engine: `models/betting_engine.py`

```python
"""
UmaEdge — Betting Analysis, Pricing & Staking Engine.
Handles fair odds calculation, verdict labeling, joint multi-finish
probability modeling (Harville/Henery), and discrete portfolio ticket construction.
"""

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class HorsePricing:
    post_position: int
    horse_name: str
    horse_name_jp: Optional[str]
    win_prob: float
    fair_odds: float
    market_odds: float
    ev: float
    verdict: str
    is_favorite: bool
    is_top_value: bool


@dataclass
class BetTicket:
    ticket_type: str        # '単勝', '複勝', 'ワイド', '馬連', '馬単', '三連複', '三連単'
    ticket_type_en: str     # 'win', 'place', 'wide', 'quinella', 'exacta', 'trio', 'trifecta'
    selection: List[int]    # List of post positions e.g. [18] or [6, 18] or [6, 9, 18]
    selection_display: str  # Formatted string e.g. "No. 18 単勝", "6–18 ワイド", "6–9–18 三連複"
    prob: float             # Joint probability of this ticket hitting
    fair_odds: float        # 1 / prob
    market_odds: Optional[float]  # Estimated or scraped market odds
    ev: Optional[float]
    stake: int              # Yen amount (multiple of 100)
    weight_pct: float       # Stake percentage


@dataclass
class StakingPlan:
    budget: int
    simple_bet: BetTicket
    portfolio_tickets: List[BetTicket]
    anchor_horse: HorsePricing
    verdict_summary: str


# ---------------------------------------------------------------------------
# 1. Pricing & Verdict Engine
# ---------------------------------------------------------------------------

def calculate_pricing_breakdown(entries: List[dict]) -> List[HorsePricing]:
    """
    Given a list of race entries with 'win_prob' and 'odds' (or 'odds_win'),
    computes fair odds, EV, and assigns plain-English verdicts.
    """
    if not entries:
        return []

    # Sort entries by win_prob descending to identify favorite
    sorted_by_prob = sorted(entries, key=lambda x: x.get("win_prob", 0.0), reverse=True)
    max_prob = sorted_by_prob[0].get("win_prob", 0.0) if sorted_by_prob else 0.0

    # Calculate EV for each entry
    priced_list = []
    for e in entries:
        prob = max(0.0001, float(e.get("win_prob") or 0.0))
        mkt_odds = float(e.get("odds") or e.get("odds_win") or 1.0)
        fair_odds = round(1.0 / prob, 1)
        ev = (prob * mkt_odds) - 1.0

        priced_list.append({
            "entry": e,
            "prob": prob,
            "fair_odds": fair_odds,
            "mkt_odds": mkt_odds,
            "ev": ev,
            "pp": e.get("post_position", 0),
            "name": e.get("horse_name", ""),
            "name_jp": e.get("horse_name_jp", ""),
        })

    # Find the top value horse (minimum 4% win prob to avoid extreme noise)
    val_candidates = [p for p in priced_list if p["prob"] >= 0.04]
    top_val_item = max(val_candidates, key=lambda x: x["ev"]) if val_candidates else priced_list[0]
    top_val_pp = top_val_item["pp"] if top_val_item["ev"] > 0.05 else None

    # Assign verdicts
    results: List[HorsePricing] = []
    for item in priced_list:
        prob = item["prob"]
        ev = item["ev"]
        pp = item["pp"]
        is_fav = (prob == max_prob and prob > 0.12)
        is_top_val = (pp == top_val_pp)

        if is_top_val and is_fav:
            verdict = "Top pick & clear value"
        elif is_top_val:
            verdict = "Best value"
        elif is_fav and ev < 0:
            verdict = "Most likely, but underpriced"
        elif ev >= 0.12 and prob >= 0.05:
            verdict = "Secondary value"
        elif prob < 0.05 and ev >= 0.25 and item["mkt_odds"] >= 20.0:
            verdict = "Longshot overlay"
        elif -0.05 <= ev < 0.10:
            verdict = "Roughly fair"
        elif -0.20 <= ev < -0.05:
            verdict = "Slightly short"
        else:
            verdict = "Clearly too short"

        results.append(HorsePricing(
            post_position=pp,
            horse_name=item["name"],
            horse_name_jp=item["name_jp"],
            win_prob=prob,
            fair_odds=item["fair_odds"],
            market_odds=item["mkt_odds"],
            ev=ev,
            verdict=verdict,
            is_favorite=is_fav,
            is_top_value=is_top_val,
        ))

    # Sort results by win_prob descending
    results.sort(key=lambda x: x.win_prob, reverse=True)
    return results


# ---------------------------------------------------------------------------
# 2. Joint Probability Modeling (Harville / Plackett-Luce)
# ---------------------------------------------------------------------------

class JointFinishModel:
    """Calculates multi-runner finishing probabilities."""

    def __init__(self, probs_by_pp: Dict[int, float], gamma: float = 0.92):
        self.probs = probs_by_pp
        self.gamma = gamma
        self.total_p = sum(probs_by_pp.values()) or 1.0

    def exact_1_2(self, i: int, j: int) -> float:
        """P(i finishes 1st AND j finishes 2nd)"""
        pi = self.probs.get(i, 0.0)
        pj = self.probs.get(j, 0.0)
        if pi <= 0 or pj <= 0 or i == j:
            return 0.0
        denom = sum(p**self.gamma for k, p in self.probs.items() if k != i)
        return pi * ((pj**self.gamma) / denom) if denom > 0 else 0.0

    def exact_1_2_3(self, i: int, j: int, k: int) -> float:
        """P(i finishes 1st AND j 2nd AND k 3rd)"""
        pi = self.probs.get(i, 0.0)
        pj = self.probs.get(j, 0.0)
        pk = self.probs.get(k, 0.0)
        if pi <= 0 or pj <= 0 or pk <= 0 or len({i, j, k}) < 3:
            return 0.0
        denom1 = sum(p**self.gamma for m, p in self.probs.items() if m != i)
        denom2 = sum(p**self.gamma for m, p in self.probs.items() if m not in (i, j))
        if denom1 <= 0 or denom2 <= 0:
            return 0.0
        return pi * ((pj**self.gamma) / denom1) * ((pk**self.gamma) / denom2)

    def wide_prob(self, i: int, j: int) -> float:
        """P(both i and j finish in the Top 3)"""
        prob = 0.0
        runners = [r for r in self.probs.keys() if r not in (i, j)]
        # Sum all permutations where i and j are in top 3
        for k in runners:
            # i 1st, j 2nd, k 3rd
            prob += self.exact_1_2_3(i, j, k)
            # i 1st, k 2nd, j 3rd
            prob += self.exact_1_2_3(i, k, j)
            # j 1st, i 2nd, k 3rd
            prob += self.exact_1_2_3(j, i, k)
            # j 1st, k 2nd, i 3rd
            prob += self.exact_1_2_3(j, k, i)
            # k 1st, i 2nd, j 3rd
            prob += self.exact_1_2_3(k, i, j)
            # k 1st, j 2nd, i 3rd
            prob += self.exact_1_2_3(k, j, i)
        return prob

    def quinella_prob(self, i: int, j: int) -> float:
        """P(i and j finish 1st and 2nd in either order)"""
        return self.exact_1_2(i, j) + self.exact_1_2(j, i)

    def trio_prob(self, i: int, j: int, k: int) -> float:
        """P(i, j, k finish in Top 3 in any order)"""
        return sum(self.exact_1_2_3(p[0], p[1], p[2]) for p in permutations([i, j, k]))


# ---------------------------------------------------------------------------
# 3. Discrete Portfolio Staking Engine
# ---------------------------------------------------------------------------

def construct_staking_plan(
    pricing: List[HorsePricing],
    budget: int = 1000,
) -> StakingPlan:
    """
    Constructs an optimal multi-ticket portfolio given horse pricing and total budget.
    Ensures all bets are multiples of ¥100 and sum exactly to budget.
    """
    if not pricing:
        raise ValueError("Pricing list cannot be empty")

    probs_by_pp = {h.post_position: h.win_prob for h in pricing}
    model = JointFinishModel(probs_by_pp)

    # 1. Identify key actors
    # Top Value horse (anchor)
    value_horses = [h for h in pricing if h.is_top_value]
    anchor = value_horses[0] if value_horses else pricing[0]

    # Favorite
    fav_horses = [h for h in pricing if h.is_favorite]
    favorite = fav_horses[0] if fav_horses else pricing[0]

    # Contenders (excluding anchor)
    contenders = [h for h in pricing if h.post_position != anchor.post_position]
    c1 = contenders[0] if len(contenders) > 0 else anchor
    c2 = contenders[1] if len(contenders) > 1 else c1

    # 2. Define Ticket Portfolio Candidates
    raw_tickets = []

    # Ticket 1: Primary Win Bet (50-60%)
    raw_tickets.append({
        "type": "単勝", "type_en": "win",
        "selection": [anchor.post_position],
        "display": f"No. {anchor.post_position} 単勝",
        "prob": anchor.win_prob,
        "weight": 0.60,
    })

    # Ticket 2: Key - Favorite Wide (20%)
    if anchor.post_position != favorite.post_position:
        w_prob = model.wide_prob(anchor.post_position, favorite.post_position)
        raw_tickets.append({
            "type": "ワイド", "type_en": "wide",
            "selection": sorted([anchor.post_position, favorite.post_position]),
            "display": f"{min(anchor.post_position, favorite.post_position)}–{max(anchor.post_position, favorite.post_position)} ワイド",
            "prob": w_prob,
            "weight": 0.20,
        })

    # Ticket 3: Key - Secondary Contender Wide (10%)
    other_c = c1 if c1.post_position != favorite.post_position else c2
    if other_c.post_position != anchor.post_position:
        w_prob2 = model.wide_prob(anchor.post_position, other_c.post_position)
        raw_tickets.append({
            "type": "ワイド", "type_en": "wide",
            "selection": sorted([anchor.post_position, other_c.post_position]),
            "display": f"{min(anchor.post_position, other_c.post_position)}–{max(anchor.post_position, other_c.post_position)} ワイド",
            "prob": w_prob2,
            "weight": 0.10,
        })

    # Ticket 4: Trio (三連複) Combination (10%)
    trio_pps = sorted(list({anchor.post_position, favorite.post_position, other_c.post_position}))
    if len(trio_pps) == 3:
        t_prob = model.trio_prob(trio_pps[0], trio_pps[1], trio_pps[2])
        raw_tickets.append({
            "type": "三連複", "type_en": "trio",
            "selection": trio_pps,
            "display": f"{trio_pps[0]}–{trio_pps[1]}–{trio_pps[2]} 三連複",
            "prob": t_prob,
            "weight": 0.10,
        })

    # 3. Discrete ¥100 Allocation Algorithm
    total_weight = sum(t["weight"] for t in raw_tickets)
    allocated_tickets: List[BetTicket] = []
    current_total = 0

    for t in raw_tickets:
        normalized_w = t["weight"] / total_weight
        raw_stake = budget * normalized_w
        # Floor to ¥100 unit
        stake = int(math.floor(raw_stake / 100.0) * 100)
        stake = max(100, stake)  # Ensure minimum ¥100 per included ticket
        current_total += stake

        allocated_tickets.append(BetTicket(
            ticket_type=t["type"],
            ticket_type_en=t["type_en"],
            selection=t["selection"],
            selection_display=t["display"],
            prob=t["prob"],
            fair_odds=round(1.0 / max(0.0001, t["prob"]), 1),
            market_odds=None,
            ev=None,
            stake=stake,
            weight_pct=normalized_w,
        ))

    # Adjust rounding differences to match exact budget
    diff = budget - sum(t.stake for t in allocated_tickets)
    if diff != 0 and allocated_tickets:
        # Add or subtract difference to the primary win ticket
        allocated_tickets[0].stake += diff

    # Simple bet: 100% on the single best pick
    simple_bet = BetTicket(
        ticket_type="単勝",
        ticket_type_en="win",
        selection=[anchor.post_position],
        selection_display=f"¥{budget:,} on No. {anchor.post_position} 単勝",
        prob=anchor.win_prob,
        fair_odds=anchor.fair_odds,
        market_odds=anchor.market_odds,
        ev=anchor.ev,
        stake=budget,
        weight_pct=1.0,
    )

    return StakingPlan(
        budget=budget,
        simple_bet=simple_bet,
        portfolio_tickets=allocated_tickets,
        anchor_horse=anchor,
        verdict_summary=f"My bet: No. {anchor.post_position} {anchor.horse_name_jp or anchor.horse_name} to win",
    )
```

---

## 6. API Response Data Contract

`GET /api/races/{race_id}/betting-analysis?budget=1000`

```json
{
  "race_id": 5503,
  "race_name": "Hanshin 11R — Osaka-Hamburg Cup",
  "budget": 1000,
  "headline": {
    "recommended_horse": "No. 18 タガノアラリア",
    "post_position": 18,
    "action": "to win",
    "summary": "My bet: No. 18 タガノアラリア to win"
  },
  "staking_plan": {
    "budget": 1000,
    "tickets": [
      {
        "type": "単勝",
        "selection": [18],
        "label": "¥600 — No. 18 単勝",
        "stake": 600,
        "prob": 0.19,
        "fair_odds": 5.3
      },
      {
        "type": "ワイド",
        "selection": [6, 18],
        "label": "¥200 — 6–18 ワイド",
        "stake": 200,
        "prob": 0.38,
        "fair_odds": 2.6
      },
      {
        "type": "ワイド",
        "selection": [9, 18],
        "label": "¥100 — 9–18 ワイド",
        "stake": 100,
        "prob": 0.22,
        "fair_odds": 4.5
      },
      {
        "type": "三連複",
        "selection": [6, 9, 18],
        "label": "¥100 — 6–9–18 三連複",
        "stake": 100,
        "prob": 0.08,
        "fair_odds": 12.5
      }
    ],
    "simple_bet_label": "If you only want one uncomplicated bet: ¥1,000 on No. 18 単勝."
  },
  "pricing_table": [
    {
      "post_position": 18,
      "horse_name": "タガノアラリア",
      "win_prob_pct": 19,
      "fair_odds": 5.3,
      "market_odds": 5.8,
      "verdict": "Best value"
    },
    {
      "post_position": 6,
      "horse_name": "セフィロ",
      "win_prob_pct": 22,
      "fair_odds": 4.5,
      "market_odds": 4.0,
      "verdict": "Most likely, but underpriced"
    },
    {
      "post_position": 9,
      "horse_name": "クルゼイロドスル",
      "win_prob_pct": 10,
      "fair_odds": 10.0,
      "market_odds": 9.7,
      "verdict": "Roughly fair"
    },
    {
      "post_position": 8,
      "horse_name": "タガノエルピーダ",
      "win_prob_pct": 9,
      "fair_odds": 11.1,
      "market_odds": 9.5,
      "verdict": "Slightly short"
    },
    {
      "post_position": 3,
      "horse_name": "ヤブサメ",
      "win_prob_pct": 8,
      "fair_odds": 12.5,
      "market_odds": 7.5,
      "verdict": "Clearly too short"
    },
    {
      "post_position": 14,
      "horse_name": "ランフォーヴァウ",
      "win_prob_pct": 8,
      "fair_odds": 12.5,
      "market_odds": 10.8,
      "verdict": "Slightly short"
    },
    {
      "post_position": 16,
      "horse_name": "ショウナンザナドゥ",
      "win_prob_pct": 7,
      "fair_odds": 14.3,
      "market_odds": 14.2,
      "verdict": "Fair"
    }
  ]
}
```

---

## 7. Frontend Visual Presentation Architecture

### 7.1. Component Layout
1. **Executive Staking Card (`.staking-card`)**:
   - Header with bold `#PP Horse Name to win`.
   - Staking breakdown bullet list (`¥600 — No. 18 単勝`, `¥200 — 6–18 ワイド`, etc.).
   - Simple bet fallback note.
2. **My Pricing Card (`.pricing-table-card`)**:
   - Table columns: `Horse` | `Estimated win chance` | `Fair odds` | `Screenshot odds` | `Verdict`.
   - Interactive badge styling for verdicts (`Best value` in emerald green, `Most likely, but underpriced` in muted purple, `Clearly too short` in muted red).
3. **Full Details (Collapsible Drawer)**:
   - Contains the existing granular data: Post position, Popularity, Model Prob bar, Market Prob, EV%, L3F time, Finish position.

---

## 8. Verification & Test Plan

1. **Unit Test: Pricing Engine (`tests/test_pricing.py`)**:
   - Verify Fair Odds $1/p$ math for boundary conditions ($p=0.001$, $p=0.50$, $p=0.99$).
   - Verify Verdict rules (Favorite with negative EV gets "Most likely, but underpriced"; Underdog with EV > 10% gets "Best value").
2. **Unit Test: Joint Finish Model (`tests/test_joint_model.py`)**:
   - Test Harville and Henery discount permutations.
   - Verify that $\sum \text{Quinella probabilities} \approx 1.0$ and Wide probabilities satisfy logical bounds.
3. **Unit Test: Discrete Staking Plan (`tests/test_staking.py`)**:
   - Test that $\sum \text{stakes} == \text{budget}$ across budgets: ¥500, ¥1,000, ¥5,000, ¥10,000.
   - Verify all stakes are exact multiples of 100.
4. **End-to-End API Test**:
   - Query `/api/races/{race_id}/betting-analysis` and assert full JSON contract.
