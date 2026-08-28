"""
UmaEdge — Betting Analysis, Pricing & Multi-Ticket Staking Engine.

Handles fair odds calculation, verdict labeling, joint multi-finish
probability modeling (Harville / Plackett-Luce with Henery discount),
and discrete portfolio ticket construction for Japanese horse racing (JRA).
"""

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple
import math
import logging
from sqlalchemy import text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EV Thresholds — single source of truth used across all engine functions.
# ---------------------------------------------------------------------------

# Minimum EV for a horse to be flagged is_top_value (and eligible as anchor).
EV_TOP_VALUE: float = 0.05

# Minimum EV for the adaptive strategy to recommend a bet rather than passing.
EV_PASS: float = 0.08

# Minimum EV for an exotic (non-win) ticket to be included in the portfolio.
# Set to 0.05 (5%) to avoid placing exotic bets whose edge sits inside the
# ±5-10% estimation error from deriving exotic odds via Harville on win pools.
# (Audit FLAW-3: EV_EXOTIC=0.0 was effectively a noise-level filter.)
# (Audit R7-SIG-1): filter uses >= so ev==EV_EXOTIC (5%) is included, matching
# the docstring intent of "minimum 5%".
EV_EXOTIC: float = 0.05

# Minimum EV for the "Best value" verdict label (top-value pick, clear edge).
EV_BEST_VALUE: float = 0.10

# Minimum EV for "Secondary value" label (non-top-value runner with positive edge).
EV_SECONDARY_VALUE: float = 0.12

# EV boundary separating "Slightly short" from "Roughly fair".
EV_SLIGHTLY_SHORT: float = -0.05

# EV boundary separating "Clearly too short" from "Slightly short".
EV_TOO_SHORT: float = -0.20

# Minimum joint finish probability for an exotic ticket to be recommended.
# Below this threshold, even high EV exotic bets are unreliable noise
# (e.g., joint prob=0.001% × market_odds=1000x → spuriously "positive" EV).
MIN_EXOTIC_JOINT_PROB: float = 0.005  # 0.5%

# Minimum individual win probability for a horse to be used as an exotic partner.
# Guards against pairing the anchor with a near-scratch horse.
MIN_EXOTIC_PARTNER_PROB: float = 0.02  # 2%


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HorsePricing:
    post_position: int
    horse_name: str
    horse_name_jp: Optional[str]
    win_prob: float
    fair_odds: float
    market_odds: Optional[float]          # None before JRA betting opens
    ev: Optional[float]                   # None when market_odds is None
    verdict: str
    is_favorite: bool          # Model-probability favorite (argmax win_prob)
    is_top_value: bool         # Highest EV horse with prob >= 4% and EV > EV_TOP_VALUE
    is_market_favorite: bool = False  # Market-odds favorite (argmin market_odds)

    @property
    def win_prob_pct(self) -> int:
        return int(round(self.win_prob * 100))

    def to_dict(self) -> dict:
        return {
            "post_position": self.post_position,
            "horse_name": self.horse_name_jp or self.horse_name,
            "horse_name_en": self.horse_name,
            "win_prob": round(self.win_prob, 4),
            "win_prob_pct": self.win_prob_pct,
            "fair_odds": self.fair_odds,
            "market_odds": self.market_odds,
            # (Audit R6-CRASH-1): ev is Optional[float]; guard against None.
            "ev": round(self.ev, 4) if self.ev is not None else None,
            "verdict": self.verdict,
            "is_favorite": self.is_favorite,
            "is_top_value": self.is_top_value,
            "is_market_favorite": self.is_market_favorite,
        }


@dataclass
class BetTicket:
    ticket_type: str        # '単勝', '複勝', 'ワイド', '馬連', '馬単', '三連複', '三連単'
    ticket_type_en: str     # 'win', 'place', 'wide', 'quinella', 'exacta', 'trio', 'trifecta'
    selection: List[int]    # List of post positions e.g. [18] or [6, 18] or [6, 9, 18]
    selection_display: str  # Formatted string e.g. "No. 18 単勝", "6–18 ワイド", "6–9–18 三連複"
    prob: float             # Joint probability of this ticket hitting
    # (Audit R7-SIG-2): Optional[float] — PASS tickets have no valid fair odds (1/0 is
    # undefined); None is semantically correct and consistent with market_odds/ev.
    fair_odds: Optional[float] = None  # 1 / prob when well-defined; None for PASS/PENDING
    market_odds: Optional[float] = None
    ev: Optional[float] = None
    stake: int = 0          # Yen amount (multiple of 100)
    weight_pct: float = 0.0 # Stake percentage
    label: str = ""         # "¥600 — No. 18 単勝"

    def __post_init__(self):
        if not self.label and self.stake > 0:
            self.label = f"¥{self.stake:,} — {self.selection_display}"

    def to_dict(self) -> dict:
        return {
            "type": self.ticket_type,
            "type_en": self.ticket_type_en,
            "selection": self.selection,
            "selection_display": self.selection_display,
            "label": self.label or f"¥{self.stake:,} — {self.selection_display}",
            "stake": self.stake,
            "prob": round(self.prob, 4),
            "fair_odds": self.fair_odds,  # None for PASS/PENDING tickets
            "market_odds": self.market_odds,
            "ev": round(self.ev, 4) if self.ev is not None else None,
            "weight_pct": round(self.weight_pct, 4),
        }


@dataclass
class StakingPlan:
    budget: int
    simple_bet: BetTicket
    portfolio_tickets: List[BetTicket]
    anchor_horse: HorsePricing
    verdict_summary: str
    simple_bet_label: str
    resolved_mode: str = "unknown"  # Actual strategy executed (differs from requested when mode="auto")

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "tickets": [t.to_dict() for t in self.portfolio_tickets],
            "simple_bet": self.simple_bet.to_dict(),
            "simple_bet_label": self.simple_bet_label,
            "anchor_horse": self.anchor_horse.to_dict(),
            "verdict_summary": self.verdict_summary,
            # (Audit R6-DICT-1): Expose resolved_mode so callers using to_dict() can
            # determine which strategy was actually executed (differs from requested
            # when strategy_mode='auto' or when a race is passed/pending).
            "resolved_mode": self.resolved_mode,
        }


# ---------------------------------------------------------------------------
# 1. Pricing & Verdict Engine
# ---------------------------------------------------------------------------

def calculate_pricing_breakdown(entries: List[dict]) -> List[HorsePricing]:
    """
    Given a list of race entries with 'win_prob' (or 'prob') and 'odds' (or 'odds_win'),
    computes fair decimal odds, EV, and assigns plain-English verdicts according to
    Section 3 of the specification.

    Win probabilities are renormalized to sum to 1.0 before EV is computed so that
    small calibration drift in the upstream ML model does not inflate or deflate EV.
    """
    if not entries:
        return []

    # ---- Step 1: Collect raw probabilities and market odds ----
    raw_probs = []
    for e in entries:
        raw_prob = float(
            e.get("win_prob") if e.get("win_prob") is not None else e.get("prob", 0.0) or 0.0
        )
        raw_probs.append(max(0.0, raw_prob))

    # Fix #6: Renormalize win probabilities so they sum to 1.0.
    total_raw_prob = sum(raw_probs)
    if total_raw_prob > 0 and abs(total_raw_prob - 1.0) > 0.001:
        norm_probs = [p / total_raw_prob for p in raw_probs]
    elif total_raw_prob > 0:
        norm_probs = raw_probs
    else:
        norm_probs = [1.0 / len(entries)] * len(entries)

    # Check if real market odds exist (> 1.0) for at least one horse.
    has_market_odds = any(
        (e.get("odds") is not None and float(e.get("odds") or 0) > 1.0)
        or (e.get("odds_win") is not None and float(e.get("odds_win") or 0) > 1.0)
        for e in entries
    )

    priced_list = []
    for idx, e in enumerate(entries):
        prob = max(0.0001, norm_probs[idx])
        fair_odds = round(1.0 / prob, 1)

        raw_odds = e.get("odds") if e.get("odds") is not None else e.get("odds_win")
        if raw_odds is not None and float(raw_odds) > 1.0:
            mkt_odds = float(raw_odds)
            ev = (prob * mkt_odds) - 1.0
        else:
            mkt_odds = None
            ev = None

        priced_list.append({
            "entry": e,
            "prob": prob,
            "raw_prob": raw_probs[idx],
            "fair_odds": fair_odds,
            "mkt_odds": mkt_odds,
            "ev": ev,
            "pp": int(e.get("post_position") or 0),
            "name": str(e.get("horse_name") or ""),
            "name_jp": e.get("horse_name_jp"),
        })

    # ---- Step 2: Identify model-probability favorite ----
    sorted_by_prob = sorted(priced_list, key=lambda x: x["prob"], reverse=True)
    max_prob = sorted_by_prob[0]["prob"] if sorted_by_prob else 0.0
    # Threshold lowered from 0.12 → 0.06 so the favourite is still identified in
    # wide-open 18-horse handicaps where no single runner exceeds 12%.
    # (Audit MINOR-1: 0.12 caused is_fav=False for all horses in large fields.)
    fav_pp = sorted_by_prob[0]["pp"] if sorted_by_prob and max_prob > 0.06 else None

    # ---- Step 3: Identify market-odds favorite ----
    mkt_quoted = [p for p in priced_list if p["mkt_odds"] is not None]
    if mkt_quoted:
        mkt_fav_item = min(mkt_quoted, key=lambda x: x["mkt_odds"])
        mkt_fav_pp = mkt_fav_item["pp"]
    else:
        mkt_fav_pp = None

    # ---- Step 4: Identify top-value horse (EV > EV_TOP_VALUE, prob >= 4%) ----
    if has_market_odds:
        val_candidates = [p for p in priced_list if p["prob"] >= 0.04 and p["ev"] is not None]
        if val_candidates:
            top_val_item = max(val_candidates, key=lambda x: x["ev"])
        else:
            top_val_item = max(
                [p for p in priced_list if p["ev"] is not None],
                key=lambda x: x["ev"],
                default=None,
            )
        top_val_pp = (
            top_val_item["pp"]
            if top_val_item and top_val_item["ev"] is not None and top_val_item["ev"] > EV_TOP_VALUE
            else None
        )
    else:
        top_val_pp = None

    # ---- Step 5: Assign verdicts ----
    results: List[HorsePricing] = []
    for rank_idx, item in enumerate(sorted_by_prob):
        prob = item["prob"]
        ev = item["ev"]
        pp = item["pp"]
        fair_odds = item["fair_odds"]
        is_fav = (pp == fav_pp) if fav_pp is not None else False
        is_top_val = (pp == top_val_pp) if top_val_pp is not None else False
        is_mkt_fav = (pp == mkt_fav_pp) if mkt_fav_pp is not None else False

        if not has_market_odds or ev is None:
            # Pre-market / unquoted verdicts based on model win chance.
            if rank_idx == 0:
                verdict = f"Model Top Pick (Fair: {fair_odds}x)"
            elif rank_idx in (1, 2):
                verdict = f"Contender (Fair: {fair_odds}x)"
            elif rank_idx in (3, 4, 5):
                verdict = f"Mid-tier (Fair: {fair_odds}x)"
            else:
                verdict = f"Longshot (Fair: {fair_odds}x)"
        else:
            # ---- Verdict decision tree ----
            # (Audit R13-MINOR-1): When is_top_val=True but the horse is a genuine
            # longshot (prob < 5%, mkt_odds >= 20x, ev >= 25%), the is_top_val branches
            # below fire BEFORE the standalone longshot_overlay branch (line ~316),
            # making "Longshot overlay" structurally unreachable for the top-EV pick.
            # A 4–5% probability horse with massive odds is a speculative play, not a
            # "Best value" pick in the confidence sense that label implies.
            # Guard at the TOP of the is_top_val cluster: longshot profile takes
            # priority over "Best value"/"Moderate value" when the criteria are met.
            _is_longshot_overlay = (
                prob < 0.05
                and ev is not None and ev >= 0.25
                and item["mkt_odds"] is not None and item["mkt_odds"] >= 20.0
            )
            if is_top_val and is_fav and ev >= EV_BEST_VALUE and not _is_longshot_overlay:
                verdict = "Top pick & clear value"
            elif is_top_val and ev >= EV_BEST_VALUE and not _is_longshot_overlay:
                verdict = "Best value"
            # (Audit NEW-INTEGRITY-2): Horse is both model favourite AND top value
            # with moderate (5–9%) EV — deserves a differentiated label so the is_fav
            # signal is not silently swallowed by the generic "Moderate value" branch.
            # (Audit R4-LOGIC-5): use >= so ev==EV_TOP_VALUE (5%) is "moderate",
            # not "slight".  The strict > boundary was an off-by-one edge case.
            # (Audit R13-MINOR-1): Also guard with _is_longshot_overlay so a sub-5%-prob
            # horse in the moderate EV range (5–9%) falls through to longshot_overlay too.
            elif is_top_val and is_fav and ev >= EV_TOP_VALUE and not _is_longshot_overlay:
                verdict = "Top pick & moderate value"
            elif is_top_val and ev >= EV_TOP_VALUE and not _is_longshot_overlay:
                verdict = "Moderate value"
            # (Audit R10-DEAD-1/2): The original 'Top pick & slight value' and
            # 'Value pick (slight edge)' branches for is_top_val + 0 < ev < EV_TOP_VALUE
            # are structurally unreachable: is_top_val=True is set only when the horse is
            # the highest-EV runner AND its ev > EV_TOP_VALUE (line 251 strict >).  Any
            # is_top_val horse therefore satisfies ev >= EV_TOP_VALUE, which is caught by
            # the two branches immediately above.  The dead branches have been removed to
            # eliminate maintenance confusion; the is_fav fallback below catches the rare
            # case of a favourite whose ev is positive but below EV_TOP_VALUE.
            # (Audit R6-VERDICT-2): model-favourite with positive but non-dominant EV
            # (not is_top_val) deserves its own label so it is distinguishable from a
            # random contender with the generic "Slight value" label.
            # (Audit R7-MINOR-2): Use >= 0.0 so ev==0.0 (exactly fairly priced favourite)
            # also gets a favourite-aware label rather than falling through to the generic
            # "Roughly fair" branch where the is_fav signal is silently lost.
            elif is_fav and ev >= 0.0:
                verdict = "Top pick & slight edge" if ev > 0.0 else "Top pick, fairly priced"
            elif is_fav and ev < 0.0:  # ev < 0.0 is now the only remaining is_fav branch
                verdict = "Most likely, but underpriced"
            elif not is_top_val and ev >= EV_SECONDARY_VALUE and prob >= 0.05:
                verdict = "Secondary value"
            # (Audit R9-VERDICT-1): Longshot overlay comes before the catch-all secondary branch
            # so that extreme overlays (ev >= 25%, prob < 5%, mkt_odds >= 20x) still get the
            # more specific "Longshot overlay" label rather than the generic "Secondary value".
            elif prob < 0.05 and ev >= 0.25 and item["mkt_odds"] is not None and item["mkt_odds"] >= 20.0:
                verdict = "Longshot overlay"
            # (Audit R9-VERDICT-1): Dead zone fix — non-top-val, non-fav horse with
            # EV > EV_BEST_VALUE (> 10%) but prob < 5% previously fell through all branches
            # to "Clearly too short". This fills the gap: EV ∈ (10%, 25%) OR prob >= 5% cases
            # not caught by secondary_value (prob >= 5%) or longshot_overlay (ev >= 25%, mkt>=20x).
            elif not is_top_val and ev > EV_BEST_VALUE:
                verdict = "Secondary value"
            # (Audit R6-VERDICT-1): Changed strict < to <= so that ev == EV_BEST_VALUE
            # (0.10) is correctly labeled "Slight value" rather than falling through to
            # the else branch ("Clearly too short") — a ludicrously wrong verdict for a
            # +10% EV horse caused by the off-by-one boundary.
            elif 0.0 < ev <= EV_BEST_VALUE:
                verdict = "Slight value"
            elif EV_SLIGHTLY_SHORT <= ev <= 0.0:
                verdict = "Roughly fair"
            elif EV_TOO_SHORT <= ev < EV_SLIGHTLY_SHORT:
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
            is_market_favorite=is_mkt_fav,
        ))

    # Sort results by win_prob descending (then post_position)
    results.sort(key=lambda x: (x.win_prob, -x.post_position), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 2. Joint Probability Modeling (Harville / Plackett-Luce with Henery)
# ---------------------------------------------------------------------------

class JointFinishModel:
    """
    Calculates multi-runner finishing probabilities for exotic bets
    using the Harville / Plackett-Luce formulation with an optional
    Henery / Stern discounting parameter gamma.

    Mathematical convention (IMPORTANT):
        P(j 2nd | i 1st) ∝  p_j^gamma  /  Σ_{m≠i} p_m^gamma

    When gamma = 1.0  →  standard Harville (no correction).
    When gamma > 1.0  →  p^gamma < p for small p  →  longshots get
        proportionally LESS weight in conditional distributions, correctly
        reducing their place probability relative to Harville.
    When gamma < 1.0  →  the correction is backwards (longshot weight
        INCREASES), which was the original bug in this codebase.

    Default gamma = 1.10 applies a mild, empirically reasonable correction.
    Use gamma = 1.0 for pure Harville (no discount).
    """

    def __init__(self, probs_by_pp: Dict[int, float], gamma: float = 1.10):
        total = sum(probs_by_pp.values())
        if total <= 0:
            n = max(1, len(probs_by_pp))
            self.probs = {k: 1.0 / n for k in probs_by_pp}
        else:
            # Normalize probabilities to sum to 1.0
            self.probs = {k: max(0.0, v / total) for k, v in probs_by_pp.items()}
        self.gamma = gamma

    def exact_1_2(self, i: int, j: int) -> float:
        """P(i finishes 1st AND j finishes 2nd)"""
        pi = self.probs.get(i, 0.0)
        pj = self.probs.get(j, 0.0)
        if pi <= 0.0 or pj <= 0.0 or i == j:
            return 0.0

        denom = sum(p**self.gamma for k, p in self.probs.items() if k != i)
        if denom <= 0.0:
            return 0.0
        return pi * ((pj**self.gamma) / denom)

    def exact_1_2_3(self, i: int, j: int, k: int) -> float:
        """P(i finishes 1st AND j 2nd AND k 3rd)"""
        pi = self.probs.get(i, 0.0)
        pj = self.probs.get(j, 0.0)
        pk = self.probs.get(k, 0.0)
        if pi <= 0.0 or pj <= 0.0 or pk <= 0.0 or len({i, j, k}) < 3:
            return 0.0

        denom1 = sum(p**self.gamma for m, p in self.probs.items() if m != i)
        denom2 = sum(p**self.gamma for m, p in self.probs.items() if m not in (i, j))
        if denom1 <= 0.0 or denom2 <= 0.0:
            return 0.0
        return pi * ((pj**self.gamma) / denom1) * ((pk**self.gamma) / denom2)

    def place_prob(self, i: int, top_n: int = 3) -> float:
        """P(horse i finishes in Top N). Usually top_n=3 (or 2 if small field)."""
        pi = self.probs.get(i, 0.0)
        if pi <= 0.0:
            return 0.0

        n_runners = len(self.probs)
        if n_runners <= 1:
            return 1.0 if pi > 0 else 0.0
        if n_runners == 2 or top_n == 2:
            prob_2nd = sum(self.exact_1_2(j, i) for j in self.probs if j != i)
            return min(1.0, pi + prob_2nd)

        # For top_n == 3: P(1st) + P(2nd) + P(3rd)
        prob_2nd = sum(self.exact_1_2(j, i) for j in self.probs if j != i)
        prob_3rd = 0.0
        for j in self.probs:
            if j == i:
                continue
            for k in self.probs:
                if k in (i, j):
                    continue
                prob_3rd += self.exact_1_2_3(j, k, i)

        return min(1.0, pi + prob_2nd + prob_3rd)

    def wide_prob(self, i: int, j: int) -> float:
        """P(both i and j finish in the Top 3)"""
        if i == j or i not in self.probs or j not in self.probs:
            return 0.0

        runners = [r for r in self.probs.keys() if r not in (i, j)]
        if not runners:
            # 2 runners total: both are top 2
            return self.quinella_prob(i, j)

        prob = 0.0
        for k in runners:
            # All 6 permutations of (i, j, k) where both i and j are in top 3
            prob += self.exact_1_2_3(i, j, k)
            prob += self.exact_1_2_3(i, k, j)
            prob += self.exact_1_2_3(j, i, k)
            prob += self.exact_1_2_3(j, k, i)
            prob += self.exact_1_2_3(k, i, j)
            prob += self.exact_1_2_3(k, j, i)
        return min(1.0, prob)

    def quinella_prob(self, i: int, j: int) -> float:
        """P(i and j finish 1st and 2nd in either order)"""
        if i == j:
            return 0.0
        return self.exact_1_2(i, j) + self.exact_1_2(j, i)

    def exacta_prob(self, i: int, j: int) -> float:
        """P(i 1st and j 2nd)"""
        return self.exact_1_2(i, j)

    def trio_prob(self, i: int, j: int, k: int) -> float:
        """P(i, j, k finish in Top 3 in any order)"""
        if len({i, j, k}) < 3:
            return 0.0
        return sum(self.exact_1_2_3(p[0], p[1], p[2]) for p in permutations([i, j, k]))

    def trifecta_prob(self, i: int, j: int, k: int) -> float:
        """P(i 1st, j 2nd, k 3rd exact order)"""
        return self.exact_1_2_3(i, j, k)


# ---------------------------------------------------------------------------
# 3. Discrete Portfolio Staking Engine
# ---------------------------------------------------------------------------

def select_adaptive_strategy(pricing: List[HorsePricing]) -> Tuple[str, str]:
    """
    Dynamically diagnoses the race value landscape and selects the optimal
    betting strategy mode ('pure_win', 'dutching', 'hybrid', or 'pass').

    Decision Rules:
    1. Pass: If top EV < 0.08 or win_prob < 0.04, return ('pass', 'No sufficient mathematical edge').
    2. Dual Dutching: If >= 2 value runners exist with EV >= 0.08, win_prob >= 0.05,
       return ('dutching', ...).
       NOTE (Audit R6-MINOR-2): construct_staking_plan uses a wider 5% (EV_TOP_VALUE)
       pool for Dutch detection — intentionally different: this selector uses the
       stricter 8% gate to avoid over-recommending dutching on marginal edges; the
       staking plan uses the looser 5% gate as a structural fallback when the caller
       already requested 'dutching' or 'hybrid' mode explicitly.
    3. Vulnerable Favorite / Exotic Fade (Hybrid): If market favorite is heavily
       overbet (EV < -0.25) and top value has EV >= 0.12, return ('hybrid', ...).
    4. Dominant Value Pick (Pure Win): 100% win allocation on top value pick
       to avoid the higher 25% exotics takeout.
    """
    if not pricing:
        return "pass", "Empty field"

    has_market_odds = any(h.market_odds is not None and h.market_odds > 1.0 and h.ev is not None for h in pricing)
    if not has_market_odds:
        return "pass", "Awaiting official JRA market odds (betting opens Friday evening). Fair odds calculated."

    # Sort by EV descending so top_v is the BEST-EV horse, not the highest-prob horse.
    # (Audit FLAW-1: pricing is ordered by win_prob, so value_horses[0] was picking
    # the highest-probability qualifying runner rather than the highest-EV one.)
    value_horses = sorted(
        [h for h in pricing if (h.is_top_value or (h.ev is not None and h.ev >= (EV_PASS - 1e-4))) and h.win_prob >= 0.04],
        key=lambda h: h.ev if h.ev is not None else -999,
        reverse=True,
    )
    fav_horses = [h for h in pricing if h.is_favorite]
    favorite = fav_horses[0] if fav_horses else pricing[0]
    top_v = value_horses[0] if value_horses else pricing[0]

    # Rule 1: Strict EV Hurdle (Pass low edges to beat 20% vig) & Min Win Prob (4%)
    if top_v.ev is None or top_v.ev < (EV_PASS - 1e-4):
        ev_str = f"{top_v.ev:+.1%}" if top_v.ev is not None else "N/A"
        return "pass", f"Pass race — Top runner EV {ev_str} below minimum +{EV_PASS:.0%} threshold. Capital preserved."
    if top_v.win_prob < 0.04:
        return "pass", f"Pass race — Top runner win chance ({top_v.win_prob:.1%}) below 4% minimum threshold. Capital preserved."

    # Rule 2: Dual Dutching (2 solid value contenders)
    # Sort by EV so the two best-EV horses are paired, not the two highest-prob ones.
    strong_value = sorted(
        [h for h in pricing if h.ev is not None and h.ev >= EV_PASS and h.win_prob >= 0.05],
        key=lambda h: h.ev,
        reverse=True,
    )
    if len(strong_value) >= 2:
        sum_p = strong_value[0].win_prob + strong_value[1].win_prob
        pos1 = f"#{strong_value[0].post_position}" if strong_value[0].post_position > 0 else strong_value[0].horse_name
        pos2 = f"#{strong_value[1].post_position}" if strong_value[1].post_position > 0 else strong_value[1].horse_name
        return "dutching", f"Dual Value Contenders ({pos1} & {pos2}, combined win {sum_p:.0%}). Dual Dutching minimizes drawdown."

    # Rule 3: Severely Overbet Favorite with High-EV Challenger -> Hybrid Exotics
    # (Audit NEW-MINOR-2): The hard -25% cliff is a discontinuous boundary; a single
    # tick in odds can flip the entire strategy.  Given ~5% model EV estimation error,
    # values in the -20% to -30% range are statistically indistinguishable.  Apply
    # a soft threshold: only trigger when favourite EV is clearly below -25% AND the
    # top value runner's EV is clearly positive (giving ≥5% safety margins each side).
    _FAV_EV_THRESHOLD = -0.25
    _FAV_EV_HYSTERESIS = 0.05  # require 5 pp clearance below threshold to fire
    if (favorite.ev is not None
            and favorite.ev < (_FAV_EV_THRESHOLD - _FAV_EV_HYSTERESIS)
            and top_v.ev is not None and top_v.ev >= EV_SECONDARY_VALUE):
        fav_pos = f"#{favorite.post_position}" if favorite.post_position > 0 else favorite.horse_name
        top_pos = f"#{top_v.post_position}" if top_v.post_position > 0 else top_v.horse_name
        return "hybrid", f"Nominal favorite {fav_pos} is severely overbet (EV: {favorite.ev:+.1%}). Hybrid portfolio captures exotic upside with {top_pos} as value anchor."

    # Rule 4: Standard Value Pick -> Pure Win (20% takeout vs 25% on exotics)
    top_pos = f"#{top_v.post_position}" if top_v.post_position > 0 else top_v.horse_name
    ev_str = f"{top_v.ev:+.1%}" if top_v.ev is not None else "N/A"
    return "pure_win", f"Strong value pick on {top_pos} ({top_v.win_prob:.0%} win prob, EV: {ev_str}). 100% Win allocation avoids exotics pool friction."


def construct_staking_plan(
    pricing: List[HorsePricing],
    budget: int = 1000,
    strategy_mode: str = "hybrid",  # "hybrid", "pure_win", "dutching", "auto"
) -> StakingPlan:
    """
    Constructs an optimal risk-adjusted betting plan for a given budget.

    Supports 4 empirical modes:
    - 'auto': AI Dynamic Strategy Selector (picks optimal style per race)
    - 'hybrid': 75-80% Core Win + 20-25% Elite Positive-EV Exotics (Sharpe 3.02, 38% Hit Rate)
    - 'pure_win': 100% Single Top Value Win (Max Raw ROI: +81.6% - +109.2%, Sharpe 3.49)
    - 'dutching': Dual Value Win Split when 2 value runners exist (Lowest DD: 8.7%, Sharpe 3.81)
    """
    # (Audit R10-MODE-1): Validate strategy_mode early so callers get a clear error
    # instead of silently falling through to a 'pure_win'-like plan with a garbage
    # resolved_mode that breaks STRATEGY_META lookups in analyze_race_betting.
    _VALID_MODES = frozenset({"hybrid", "pure_win", "dutching", "auto", "adaptive"})
    if strategy_mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown strategy_mode={strategy_mode!r}. "
            f"Valid values are: {sorted(_VALID_MODES)}"
        )

    if not pricing:
        raise ValueError("Pricing list cannot be empty")

    has_market_odds = any(h.market_odds is not None and h.market_odds > 1.0 and h.ev is not None for h in pricing)
    if not has_market_odds:
        anchor = pricing[0]
        anchor_name = anchor.horse_name_jp or anchor.horse_name
        pos_str = f"No. {anchor.post_position} " if anchor.post_position > 0 else ""
        verdict_summary = f"⏳ Market Odds Pending — Official JRA betting opens Friday evening. Model top pick is {pos_str}{anchor_name} ({anchor.win_prob_pct}% win chance, Fair: {anchor.fair_odds}x)."
        simple_label = f"⏳ JRA betting opens Friday evening. Model fair odds: {anchor.fair_odds}x on {pos_str}{anchor_name}."
        return StakingPlan(
            budget=budget,
            simple_bet=BetTicket(
                ticket_type="PENDING",
                ticket_type_en="pending",
                selection=[anchor.post_position] if anchor.post_position > 0 else [],
                selection_display=f"{pos_str}{anchor_name} (Fair: {anchor.fair_odds}x)",
                prob=anchor.win_prob,
                fair_odds=anchor.fair_odds,
                market_odds=None,
                ev=None,
                stake=0,
                weight_pct=0.0,
                label=f"⏳ Awaiting Market Odds — {pos_str}{anchor_name}",
            ),
            portfolio_tickets=[],
            anchor_horse=anchor,
            verdict_summary=verdict_summary,
            simple_bet_label=simple_label,
            resolved_mode="pending",
        )

    # If auto / adaptive mode is requested, select optimal strategy dynamically
    if strategy_mode in ("auto", "adaptive"):
        chosen_mode, auto_reason = select_adaptive_strategy(pricing)
        if chosen_mode == "pass":
            anchor = pricing[0]
            verdict_summary = f"Pass — {auto_reason}"
            simple_label = "Pass this race — No positive mathematical edge found. Preserve bankroll."
            return StakingPlan(
                budget=budget,
                simple_bet=BetTicket(
                    ticket_type="PASS",
                    ticket_type_en="pass",
                    selection=[],
                    selection_display="PASS",
                    prob=0.0,
                    # (Audit R8-FIX-1): R7-SIG-2 missed this auto-mode PASS path.
                    # Apply the same None semantics: fair_odds and ev are undefined for a PASS.
                    fair_odds=None,
                    market_odds=None,
                    ev=None,
                    stake=0,
                    weight_pct=0.0,
                    label="Pass this race",
                ),
                portfolio_tickets=[],
                anchor_horse=anchor,
                verdict_summary=verdict_summary,
                simple_bet_label=simple_label,
                resolved_mode="pass",
            )
        strategy_mode = chosen_mode

    # (Audit INTEGRITY-3): enforce ¥100 multiple.
    # (Audit R5-LOGIC-5): use floor, not round — round() can exceed the caller's stated budget.
    # (Audit R6-LOGIC-2): raise instead of silently inflating — max(100, ...) used to exceed the
    # caller's stated budget for any input in [1, 99].  The minimum JRA bet unit is ¥100;
    # callers must pass at least ¥100 or handle the ValueError upstream.
    budget = int(budget // 100) * 100
    if budget < 100:
        raise ValueError(
            f"Budget must be at least ¥100 (minimum JRA bet unit). "
            f"Got ¥{budget} after flooring to nearest ¥100."
        )
    # (Audit R4-INTEGRITY-4): Exclude pp=0 phantom entries created when post_position
    # is None/"" in the source data (int(None or 0) → 0).  A pp=0 runner does not exist
    # in JRA but would enter JointFinishModel and pollute every conditional denominator,
    # understating all real horses' place probabilities.
    probs_by_pp = {h.post_position: h.win_prob for h in pricing if h.post_position > 0}
    # (Audit R7-MINOR-3): Use gamma=1.0 for ML-derived probabilities.  The Henery/Stern
    # gamma correction is designed to counteract market-pool longshot bias (bettors
    # over-weighting longshots in pari-mutuel pools).  ML model probs should already be
    # calibrated and do not exhibit this bias, so applying gamma>1.0 would incorrectly
    # under-weight longshots' place probability vs what the model predicts.
    # mkt_model and mkt_full_model retain gamma=1.10 since they derive from market odds.
    model = JointFinishModel(probs_by_pp, gamma=1.0)

    # 1. Identify key actors
    # --- Top Value horse (anchor) ---
    # (Audit R3-FLAW-2 + R3-LOGIC-1): Anchor must be the highest-EV runner so that
    # the Dutch pair selected here is identical to the pair selected by
    # select_adaptive_strategy (which also sorts by EV).  Using value_horses_flagged
    # first created a desynchronisation: the strategy selector said "Dutch PP5 & PP9"
    # but the staking plan bet "PP3 (is_top_value) + PP9".  Now anchor = ev_qualified[0]
    # always, with value_horses_flagged as a fallback for the no-market-odds path.
    ev_qualified = sorted(
        [h for h in pricing if h.ev is not None and h.ev >= EV_PASS and h.win_prob >= 0.05],
        key=lambda h: h.ev,
        reverse=True,
    )
    value_horses_flagged = [h for h in pricing if h.is_top_value]
    # Primary anchor: highest-EV runner that clears the EV_PASS threshold.
    # Fallback to is_top_value horse (EV_TOP_VALUE >= 5%) so we still have an anchor
    # when no horse clears 8% EV but one is the clear model top value pick.
    anchor = ev_qualified[0] if ev_qualified else (value_horses_flagged[0] if value_horses_flagged else pricing[0])

    # --- Favorite (model-probability) ---
    fav_horses = [h for h in pricing if h.is_favorite]
    favorite = fav_horses[0] if fav_horses else pricing[0]

    # --- Fix #5: Contenders sorted by EV descending so exotic tickets pair with best-value runners,
    # not the highest-probability (often overpriced) horse. ---
    contenders = sorted(
        [h for h in pricing if h.post_position != anchor.post_position],
        key=lambda h: h.ev if h.ev is not None else -999,
        reverse=True,
    )
    # --- Fix: Exotic contenders must have meaningful individual probability ---
    # After EV-sort, filter out near-scratch horses that can't realistically place.
    viable_contenders = [
        h for h in contenders if h.win_prob >= MIN_EXOTIC_PARTNER_PROB
    ]
    c1 = viable_contenders[0] if len(viable_contenders) > 0 else (contenders[0] if contenders else anchor)
    c2 = viable_contenders[1] if len(viable_contenders) > 1 else c1

    # JRA Statutory Takeout Rates
    TAKEOUT_MAP = {
        "win": 0.20,
        "place": 0.20,
        "wide": 0.225,
        "quinella": 0.225,
        "exacta": 0.25,
        "trio": 0.25,
        "trifecta": 0.275,
    }

    # Market-implied joint model for exotic pool pricing.
    # (Audit R3-FLAW-1): Only include horses with real market odds.  Substituting 999.0
    # for unquoted runners adds near-zero implied probabilities that dilute the joint
    # probability of *quoted* runners after JointFinishModel normalises; the resulting
    # joint probs are too small, making computed exotic odds appear larger than the market
    # would actually offer, inflating exotic EV in partial fields by up to 15-20%.
    # By excluding unquoted runners we preserve the correct relative odds relationships
    # among the horses that are actually being priced.
    mkt_probs_by_pp = {
        h.post_position: 1.0 / max(1.01, h.market_odds)
        for h in pricing
        if h.market_odds is not None and h.market_odds > 1.0
    }
    # Safety: if the anchor or contenders are somehow absent from mkt_probs, fall back
    # to uniform to avoid KeyError in JointFinishModel calculations.
    if not mkt_probs_by_pp:
        mkt_probs_by_pp = {h.post_position: 1.0 / max(1, len(pricing)) for h in pricing}
    mkt_model = JointFinishModel(mkt_probs_by_pp, gamma=model.gamma)

    # (Audit R4-LOGIC-2 + R4-INTEGRITY-5): Build a full-field market model for exotic
    # joint probability pricing.  mkt_model only contains quoted horses, which is correct
    # for single-horse win EV checks but wrong for joint exotic pricing: with only 3
    # quoted runners in an 18-horse field, mkt_model.wide_prob() ≈ 1.0 because
    # P(both in top 3 of a 3-runner field) ≈ 1 → wide market odds = 0.775x (nonsensical).
    # mkt_full_model includes ALL runners; unquoted ones receive a pseudo-probability =
    # half the smallest quoted implied prob so they contribute to the denominator without
    # inflating any quoted horse's win EV.
    _min_quoted_impl = min(mkt_probs_by_pp.values()) if mkt_probs_by_pp else 0.01
    _unquoted_pseudo = max(0.001, _min_quoted_impl / 2.0)
    mkt_full_probs_by_pp = dict(mkt_probs_by_pp)
    for _h in pricing:
        if _h.post_position > 0 and _h.post_position not in mkt_full_probs_by_pp:
            mkt_full_probs_by_pp[_h.post_position] = _unquoted_pseudo
    mkt_full_model = JointFinishModel(mkt_full_probs_by_pp, gamma=model.gamma)

    # (Audit R4-INTEGRITY-5): If the anchor itself has no market odds, exotic EV cannot
    # be reliably computed — the anchor's mkt_p would be the pseudo-value, not a real
    # market price, making computed exotic market_odds meaningless.  Skip all exotic
    # ticket generation in this edge case.
    anchor_in_mkt = anchor.post_position in mkt_probs_by_pp
    # (Audit R5-INTEGRITY-3): Check whether exotic partners have real (non-pseudo) market
    # odds.  If they lack real quotes, mkt_full_model assigns them pseudo-probs, producing
    # nonsensically high exotic market_odds (same class of bug as R4-INTEGRITY-5 for anchor).
    c1_in_mkt = c1.post_position in mkt_probs_by_pp
    c2_in_mkt = c2.post_position in mkt_probs_by_pp

    # (Audit R5-INTEGRITY-1 + Fix #2 CRITICAL): Dutch detection now uses a wider 5%+ EV pool
    # (EV_TOP_VALUE) rather than the 8%+ EV_PASS pool, eliminating the asymmetry where the
    # anchor could fall back to 5% EV but a second 5-7% EV runner was never considered for
    # Dutching.  We still require anchor.ev >= EV_TOP_VALUE to gate the Dutch opportunity.
    _dutch_pool = sorted(
        [h for h in pricing if h.ev is not None and h.ev >= EV_TOP_VALUE and h.win_prob >= 0.05],
        key=lambda h: h.ev,
        reverse=True,
    )
    has_dutch = False
    second_v = None
    if len(_dutch_pool) >= 2 and anchor.ev is not None and anchor.ev >= EV_TOP_VALUE:
        for h in _dutch_pool:
            if h.post_position != anchor.post_position:
                second_v = h
                break
        if second_v is not None:
            has_dutch = True
    # (Audit R5-LOGIC-3): Dutch requires second_v to have real market odds.  Without them
    # the Dutch ticket has no displayable odds and the Kelly weight uses an arbitrary 2x
    # fallback.  Clear has_dutch if second_v is unquoted.
    if has_dutch and second_v is not None:
        if second_v.market_odds is None or second_v.market_odds <= 1.0:
            has_dutch = False
            second_v = None

    # 3. Generate Candidate Tickets across styles (Win, Dutch Win, Exacta, Quinella, Wide, Trio)
    candidate_tickets = []

    # (Audit R4-LOGIC-3): Dutch split was hardcoded 65/35, ignoring relative EV.
    # Compute Kelly-proportional weights: k = EV / (odds - 1).  Since all weights
    # are normalised by total_weight later, only the ratio between the two legs
    # matters — this correctly sizes each bet proportional to its mathematical edge.
    if has_dutch and second_v is not None and strategy_mode in ("dutching", "hybrid"):
        _ak = max(0.001, (anchor.ev or 0.0) / max(0.001, (anchor.market_odds or 2.0) - 1.0))
        _sk = max(0.001, (second_v.ev or 0.0) / max(0.001, (second_v.market_odds or 2.0) - 1.0))
        _total_kelly = _ak + _sk
        primary_win_weight = _ak / _total_kelly
        dutch_weight = _sk / _total_kelly
    else:
        primary_win_weight = 0.65 if (has_dutch and strategy_mode != "pure_win") else 1.0
        dutch_weight = 0.35 if strategy_mode == "dutching" else 0.25

    # Ticket 1: Primary Win Bet (Anchor)
    candidate_tickets.append({
        "type": "単勝", "type_en": "win",
        "selection": [anchor.post_position],
        "display": f"No. {anchor.post_position} 単勝" + (" (Primary)" if (has_dutch and strategy_mode != "pure_win") else ""),
        "prob": anchor.win_prob,
        "weight": primary_win_weight,
        "market_odds": anchor.market_odds,
        "ev": anchor.ev,
        "is_core": True,
    })

    # Dutching Ticket (Secondary Value Win)
    if has_dutch and second_v is not None and strategy_mode in ("dutching", "hybrid"):
        candidate_tickets.append({
            "type": "単勝", "type_en": "win",
            "selection": [second_v.post_position],
            "display": f"No. {second_v.post_position} 単勝 (Dutch)",
            "prob": second_v.win_prob,
            "weight": dutch_weight,
            "market_odds": second_v.market_odds,
            "ev": second_v.ev,
            "is_core": True,
        })

    # Exacta Candidate (馬単: Anchor -> Contender)
    # (Audit R4-INTEGRITY-5): Gate ALL exotic tickets on anchor_in_mkt.  If the anchor
    # has no market odds, its mkt_p = 0 → exotic market_odds = 7500x → fake EV.
    # Also use mkt_full_model (full field with pseudo-probs for unquoted runners) instead
    # of mkt_model (quoted only) so that joint probs are computed against a realistic
    # full field, not just 3 quoted runners where wide_prob ≈ 1.0.
    if c1.post_position != anchor.post_position and anchor_in_mkt and c1_in_mkt:
        ex_p = model.exacta_prob(anchor.post_position, c1.post_position)
        ex_mkt_p = mkt_full_model.exacta_prob(anchor.post_position, c1.post_position)
        ex_odds = round((1.0 - TAKEOUT_MAP["exacta"]) / max(0.0001, ex_mkt_p), 1)
        ex_ev = (ex_p * ex_odds) - 1.0
        candidate_tickets.append({
            "type": "馬単", "type_en": "exacta",
            "selection": [anchor.post_position, c1.post_position],
            "display": f"{anchor.post_position} → {c1.post_position} 馬単",
            "prob": ex_p,
            "weight": 0.15,
            "market_odds": ex_odds,
            "ev": ex_ev,
            "is_core": False,
        })

    # Quinella Candidate (馬連: Anchor - Partner)
    # Use c1 (best EV-qualified contender) as partner — not the raw model-probability
    # favorite, which can be the same as the anchor or have near-zero probability.
    # (Audit NEW-FLAW-2): The MINOR-4 dedup guard that removed Quinella whenever it
    # shared the same pair as Wide has been reverted.  Quinella ⊆ Wide: Quinella pays
    # only for a 1-2 finish, Wide also covers 1-3 / 2-3.  In a 4-horse race at equal
    # win probabilities P(Wide NOT Quin) ≈ 41% — they are nested, not correlated.
    # Holding both diversifies between the two distinct payout scenarios.
    quinella_partner = c1 if c1.post_position != anchor.post_position else c2
    quinella_partner_in_mkt = quinella_partner.post_position in mkt_probs_by_pp
    if quinella_partner.post_position != anchor.post_position and anchor_in_mkt and quinella_partner_in_mkt:
        q_p = model.quinella_prob(anchor.post_position, quinella_partner.post_position)
        q_mkt_p = mkt_full_model.quinella_prob(anchor.post_position, quinella_partner.post_position)
        q_odds = round((1.0 - TAKEOUT_MAP["quinella"]) / max(0.0001, q_mkt_p), 1)
        q_ev = (q_p * q_odds) - 1.0
        candidate_tickets.append({
            "type": "馬連", "type_en": "quinella",
            "selection": sorted([anchor.post_position, quinella_partner.post_position]),
            "display": f"{min(anchor.post_position, quinella_partner.post_position)}–{max(anchor.post_position, quinella_partner.post_position)} 馬連",
            "prob": q_p,
            "weight": 0.15,
            "market_odds": q_odds,
            "ev": q_ev,
            "is_core": False,
        })

    # Wide Candidates (ワイド)
    # Use the best EV-qualified partner (c1), falling back to the model favorite
    # only if c1 IS the anchor.
    wide_partner = c1 if c1.post_position != anchor.post_position else (
        favorite if favorite.post_position != anchor.post_position else c2
    )
    wide_partner_in_mkt = wide_partner.post_position in mkt_probs_by_pp
    # (Audit NEW-MINOR-1): JRA does not offer Wide for fields with fewer than 3 runners.
    # wide_prob() returns quinella_prob=1.0 for 2-runner fields, producing a nonsense ticket.
    # (Audit R5-LOGIC-6): Raise guard to >= 4 — wide_prob on a 3-runner model also returns
    # ~1.0 (sum of all 3! permutations = 1.0), producing 0.775x market odds and negative EV.
    if wide_partner.post_position != anchor.post_position and len(pricing) >= 4 and anchor_in_mkt and wide_partner_in_mkt:
        w_p = model.wide_prob(anchor.post_position, wide_partner.post_position)
        w_mkt_p = mkt_full_model.wide_prob(anchor.post_position, wide_partner.post_position)
        w_odds = round((1.0 - TAKEOUT_MAP["wide"]) / max(0.0001, w_mkt_p), 1)
        w_ev = (w_p * w_odds) - 1.0
        candidate_tickets.append({
            "type": "ワイド", "type_en": "wide",
            "selection": sorted([anchor.post_position, wide_partner.post_position]),
            "display": f"{min(anchor.post_position, wide_partner.post_position)}–{max(anchor.post_position, wide_partner.post_position)} ワイド",
            "prob": w_p,
            "weight": 0.12,
            "market_odds": w_odds,
            "ev": w_ev,
            "is_core": False,
        })

    other_c = c1 if c1.post_position not in (anchor.post_position, wide_partner.post_position) else c2
    other_c_in_mkt = other_c.post_position in mkt_probs_by_pp
    if (other_c.post_position != anchor.post_position
            and other_c.post_position != wide_partner.post_position
            and other_c.win_prob >= MIN_EXOTIC_PARTNER_PROB
            and anchor_in_mkt and other_c_in_mkt):
        w2_p = model.wide_prob(anchor.post_position, other_c.post_position)
        w2_mkt_p = mkt_full_model.wide_prob(anchor.post_position, other_c.post_position)
        w2_odds = round((1.0 - TAKEOUT_MAP["wide"]) / max(0.0001, w2_mkt_p), 1)
        w2_ev = (w2_p * w2_odds) - 1.0
        candidate_tickets.append({
            "type": "ワイド", "type_en": "wide",
            "selection": sorted([anchor.post_position, other_c.post_position]),
            "display": f"{min(anchor.post_position, other_c.post_position)}–{max(anchor.post_position, other_c.post_position)} ワイド",
            "prob": w2_p,
            "weight": 0.10,
            "market_odds": w2_odds,
            "ev": w2_ev,
            "is_core": False,
        })

    # Trio Candidate (三連複)
    trio_pps = sorted(list({anchor.post_position, wide_partner.post_position, other_c.post_position}))
    if len(trio_pps) == 3 and anchor_in_mkt and wide_partner_in_mkt and other_c_in_mkt:
        tr_p = model.trio_prob(trio_pps[0], trio_pps[1], trio_pps[2])
        tr_mkt_p = mkt_full_model.trio_prob(trio_pps[0], trio_pps[1], trio_pps[2])
        tr_odds = round((1.0 - TAKEOUT_MAP["trio"]) / max(0.0001, tr_mkt_p), 1)
        tr_ev = (tr_p * tr_odds) - 1.0
        candidate_tickets.append({
            "type": "三連複", "type_en": "trio",
            "selection": trio_pps,
            "display": f"{trio_pps[0]}–{trio_pps[1]}–{trio_pps[2]} 三連複",
            "prob": tr_p,
            "weight": 0.08,
            "market_odds": tr_odds,
            "ev": tr_ev,
            "is_core": False,
        })

    # 4. Joint Probability Floor + Strict Positive EV Filter
    core_tickets = [t for t in candidate_tickets if t["is_core"]]
    exotic_candidates = [t for t in candidate_tickets if not t["is_core"]]

    # Sort exotics by EV descending
    exotic_candidates.sort(key=lambda x: x["ev"], reverse=True)

    qualified_exotics = []
    if strategy_mode not in ("pure_win", "dutching"):
        for ex in exotic_candidates:
            if len(qualified_exotics) >= 2:
                break
            # Joint probability floor: reject bets that are practically impossible
            # regardless of their nominal EV (EV is unreliable at tiny probabilities)
            if ex["prob"] < MIN_EXOTIC_JOINT_PROB:
                continue
            # (Audit R7-SIG-1): Use >= so ev==EV_EXOTIC (5%) is included, matching
            # the docstring intent of "minimum 5%".  The prior strict > excluded it.
            if ex["ev"] >= EV_EXOTIC:
                qualified_exotics.append(ex)

    # Check if edge exists
    # (Audit R3-INTEGRITY-3): anchor.ev can be None when the anchor horse has no market
    # odds — comparing None > 0.0 raises TypeError at runtime.  Guard explicitly.
    # (Audit R6-LOGIC-1): The core win ticket is always placed when has_positive_edge is
    # True, so the anchor MUST have positive win EV independently.  The old
    # `or bool(qualified_exotics)` clause allowed inflated Harville exotic EV (e.g.
    # sub-1.0 odds fields where implied probs sum > 1.0) to satisfy has_positive_edge
    # while placing a large negative-EV win bet as the cornerstone ticket.
    # Exotics alone cannot rescue a negative-anchor plan — return PASS instead.
    has_positive_edge = anchor.ev is not None and anchor.ev > 0.0
    anchor_name = anchor.horse_name_jp or anchor.horse_name

    # (Audit R7-CRITICAL-2): Manual modes ('hybrid', 'pure_win', 'dutching') bypass
    # select_adaptive_strategy, so the EV_PASS gate is not enforced.  A horse with
    # EV > EV_TOP_VALUE(5%) but < EV_PASS(8%) would pass has_positive_edge below and
    # trigger a real bet.  Enforce the same gate here so manual modes cannot silently
    # place sub-threshold bets.  The anchor is the best available horse; if its EV
    # doesn't clear EV_PASS the race should be passed regardless of requested mode.
    if anchor.ev is not None and 0.0 < anchor.ev < (EV_PASS - 1e-4):
        _ev_str = f"{anchor.ev:+.1%}"
        verdict_summary = f"Pass — {anchor.ev:+.1%} EV on best pick is below the minimum +{EV_PASS:.0%} threshold. Capital preserved."
        simple_label = "Pass this race — No positive mathematical edge found. Preserve bankroll."
        return StakingPlan(
            budget=budget,
            simple_bet=BetTicket(
                ticket_type="PASS",
                ticket_type_en="pass",
                selection=[],
                selection_display="PASS",
                prob=0.0,
                fair_odds=None,
                market_odds=None,
                ev=None,
                stake=0,
                weight_pct=0.0,
                label="Pass this race",
            ),
            portfolio_tickets=[],
            anchor_horse=anchor,
            verdict_summary=verdict_summary,
            simple_bet_label=simple_label,
            resolved_mode="pass",
        )

    if not has_positive_edge:
        # (Audit R4-CRASH-1): anchor.ev can be None here (that's what makes has_positive_edge
        # False).  The old format string {anchor.ev:+.1%} raised TypeError when ev is None.
        _ev_str = f"{anchor.ev:+.1%}" if anchor.ev is not None else "N/A"
        verdict_summary = f"Pass — No value bets in this race. All runners underpriced (Best EV: {_ev_str})."
        simple_label = "Pass this race — No positive mathematical edge found. Preserve bankroll."
        return StakingPlan(
            budget=budget,
            simple_bet=BetTicket(
                ticket_type="PASS",
                ticket_type_en="pass",
                selection=[],
                selection_display="PASS",
                prob=0.0,
                # (Audit R7-SIG-2): Use None for PASS tickets — fair_odds=0.0 was semantically
                # invalid (1/0 is undefined) and ev=0.0 was misleading (implies breakeven, not N/A).
                # Consistent with the 'pending' path which also uses ev=None.
                fair_odds=None,
                market_odds=None,
                ev=None,
                stake=0,
                weight_pct=0.0,
                label="Pass this race",
            ),
            portfolio_tickets=[],
            anchor_horse=anchor,
            verdict_summary=verdict_summary,
            simple_bet_label=simple_label,
            resolved_mode="pass",
        )

    if not qualified_exotics:
        raw_tickets = core_tickets
        total_core_w = sum(t["weight"] for t in core_tickets)
        for t in raw_tickets:
            t["weight"] = t["weight"] / total_core_w
    else:
        # High-Sharpe Hybrid Allocation: Core Win gets 75-80%, Exotics get 20-25%
        win_weight = 0.75 if has_dutch else 0.80
        exotic_pool_weight = 1.0 - win_weight

        total_core_w = sum(t["weight"] for t in core_tickets)
        for t in core_tickets:
            t["weight"] = win_weight * (t["weight"] / total_core_w)

        raw_tickets = list(core_tickets)

        # (Audit NEW-INTEGRITY-3): Weight = prob × (1 + EV) is a Sharpe-maximisation
        # formula — it prioritises high-probability, lower-variance exotics over high-EV
        # long-shots.  A pure EV-maximisation formula would be weight = prob × EV, but
        # that concentrates capital in tiny-probability tickets whose EV estimates are
        # unreliable.  The current choice is intentional: stable expected profit over
        # maximised single-ticket expectation.  To change this behaviour, adjust the
        # scoring formula below (e.g. set score = prob * ev for ev_max mode).
        exotic_total_score = sum(max(0.01, ex["prob"] * max(0.1, 1.0 + ex["ev"])) for ex in qualified_exotics)
        for ex in qualified_exotics:
            score = max(0.01, ex["prob"] * max(0.1, 1.0 + ex["ev"]))
            ex["weight"] = round(exotic_pool_weight * (score / exotic_total_score), 3)
            raw_tickets.append(ex)

    # 3. Discrete ¥100 Allocation Algorithm
    # If budget is limited (e.g. < ¥400), prune lower-weight tickets so every included ticket has >= ¥100
    max_tickets = max(1, budget // 100)
    if len(raw_tickets) > max_tickets:
        raw_tickets = raw_tickets[:max_tickets]

    total_weight = sum(t["weight"] for t in raw_tickets) or 1.0

    allocated_tickets: List[BetTicket] = []
    stakes = []
    remainders = []

    for t in raw_tickets:
        norm_w = t["weight"] / total_weight
        raw_stake = budget * norm_w
        floored_stake = int(math.floor(raw_stake / 100.0) * 100)
        # Ensure minimum ¥100 per ticket if budget allows
        floored_stake = max(100, floored_stake)
        stakes.append(floored_stake)
        remainders.append((raw_stake - floored_stake, len(stakes) - 1))

    # If sum of initial stakes exceeds budget (can happen with small budgets due to min ¥100):
    while sum(stakes) > budget:
        # Reduce from the lowest weight / lowest remainder ticket that is > 100
        reducible = [i for i, s in enumerate(stakes) if s > 100]
        if reducible:
            idx_to_reduce = min(reducible, key=lambda i: (raw_tickets[i]["weight"], -i))
            stakes[idx_to_reduce] -= 100
        else:
            # If all are at 100 and still over budget, pop the lowest weight ticket
            stakes.pop()
            raw_tickets.pop()

    # Re-normalize total weight if tickets were popped
    total_weight = sum(t["weight"] for t in raw_tickets) or 1.0

    # Distribute undistributed remainder balance in ¥100 units
    diff = budget - sum(stakes)
    if diff > 0:
        # Sort indices by ticket weight descending
        sorted_indices = sorted(
            range(len(raw_tickets)),
            key=lambda i: (raw_tickets[i]["weight"]),
            reverse=True,
        )
        step = 0
        while diff >= 100 and sorted_indices:
            idx = sorted_indices[step % len(sorted_indices)]
            stakes[idx] += 100
            diff -= 100
            step += 1

    # Build final BetTicket objects
    for i, t in enumerate(raw_tickets):
        stk = stakes[i]
        norm_w = t["weight"] / total_weight
        fair_odds = round(1.0 / max(0.0001, t["prob"]), 1)
        allocated_tickets.append(BetTicket(
            ticket_type=t["type"],
            ticket_type_en=t["type_en"],
            selection=t["selection"],
            selection_display=t["display"],
            prob=t["prob"],
            fair_odds=fair_odds,
            market_odds=t.get("market_odds"),
            ev=t.get("ev"),
            stake=stk,
            weight_pct=norm_w,
            label=f"¥{stk:,} — {t['display']}",
        ))

    # 4. Simple Bet Alternative (100% Win on Anchor)
    simple_bet = BetTicket(
        ticket_type="単勝",
        ticket_type_en="win",
        selection=[anchor.post_position],
        selection_display=f"No. {anchor.post_position} 単勝",
        prob=anchor.win_prob,
        fair_odds=anchor.fair_odds,
        market_odds=anchor.market_odds,
        ev=anchor.ev,
        stake=budget,
        weight_pct=1.0,
        label=f"¥{budget:,} on No. {anchor.post_position} 単勝",
    )

    # (Audit NEW-INTEGRITY-1): In dutching mode two win tickets are placed;
    # the summary must mention both horses so the user is not surprised.
    # (Audit R3-LOGIC-2): When hybrid was requested but no exotics qualified, the plan
    # silently degraded to a pure-win or dutching plan with no explanation.  Surface this.
    #
    # (Audit R11-MINOR-1): Detect whether qualifying exotics were budget-pruned.  When
    # qualified_exotics is non-empty but max_tickets truncation discarded them all, the
    # previous code showed no indication — a silent degradation parallel to the Dutch
    # truncation case fixed in R9-MINOR-1.  Detect via: no exotic ticket in allocated_tickets
    # even though qualified_exotics was non-empty.  This is the complementary disclosure.
    _n_exotic_allocated = sum(
        1 for t in allocated_tickets if t.ticket_type_en not in ("win", "place")
    )
    _exotics_budget_pruned = (
        strategy_mode == "hybrid"
        and bool(qualified_exotics)
        and _n_exotic_allocated == 0
    )

    if has_dutch and second_v is not None and strategy_mode in ("dutching", "hybrid"):
        second_name = second_v.horse_name_jp or second_v.horse_name
        dutch_suffix = ""
        if strategy_mode == "hybrid" and not qualified_exotics:
            dutch_suffix = " (no qualifying exotics — full budget to win bets)"
        elif _exotics_budget_pruned:
            # (Audit R11-MINOR-1): Exotics qualified but budget only covered win bets.
            dutch_suffix = f" (exotics available — increase budget above ¥{budget:,} for full hybrid)"
        verdict_summary = (
            f"My bets: No. {anchor.post_position} {anchor_name} & "
            f"No. {second_v.post_position} {second_name} to win (Dutch){dutch_suffix}"
        )
    else:
        win_suffix = ""
        if strategy_mode == "hybrid" and not qualified_exotics:
            win_suffix = " (no qualifying exotics — full budget to win)"
        elif _exotics_budget_pruned:
            # (Audit R11-MINOR-1): Same disclosure for single-anchor hybrid.
            win_suffix = f" (exotics available — increase budget above ¥{budget:,} for full hybrid)"
        elif strategy_mode == "pure_win" and len(ev_qualified) >= 2:
            # (Audit R4-LOGIC-4): user explicitly chose pure_win but two value runners
            # exist — the Dutch opportunity was silently suppressed.  Surface it.
            win_suffix = " (2 value runners found — switch to 'dutching' for Dual Dutch allocation)"
        elif strategy_mode == "dutching" and not has_dutch:
            # (Audit R7-MINOR-1): dutching was requested but only 1 value horse was found.
            # Degrade gracefully and surface the reason so resolved_mode is accurate.
            win_suffix = " (only 1 value runner found — dutching unavailable, single win bet placed)"
        verdict_summary = f"My bet: No. {anchor.post_position} {anchor_name} to win{win_suffix}"
    simple_bet_label = f"If you only want one uncomplicated bet: ¥{budget:,} on No. {anchor.post_position} 単勝."

    # (Audit R7-MINOR-1): When 'dutching' was requested but has_dutch=False, the actual
    # execution was pure_win.  Reflect this in resolved_mode so callers and STRATEGY_META
    # lookup show the correct executed strategy, not the requested one.
    # (Audit R9-MINOR-1): Also handle budget-truncation degradation: if strategy_mode is
    # 'dutching' but only 1 win ticket was actually allocated (due to budget//100 < 2),
    # the resolved_mode and verdict_summary must reflect the single-bet reality.
    # (Audit R11-MINOR-1): Exotic-only budget pruning does not change resolved_mode — the
    # core Dutch allocation succeeded; only the exotic overlay was dropped.  The mode
    # stays 'hybrid' and the suffix message informs the caller.
    # (Audit R12-MINOR-1): When hybrid degrades to Dutch-only (has_dutch=True, 0 exotics
    # allocated), resolved_mode stays 'hybrid' but STRATEGY_META shows "75% Core Win + 25%
    # Elite Exotics" — misleading since no exotics were placed.  Downgrade to 'dutching'.
    # (Audit R12-MINOR-2): When hybrid degrades to single-win-only (has_dutch=False, 0
    # exotics allocated), resolved_mode stays 'hybrid' while the verdict correctly says
    # "(no qualifying exotics — full budget to win)".  Downgrade to 'pure_win' so
    # STRATEGY_META matches the actual execution.
    _actual_mode = strategy_mode
    n_win_tickets = sum(1 for t in allocated_tickets if t.ticket_type_en == "win")

    if strategy_mode == "dutching" and not has_dutch:
        # has_dutch=False: only 1 value horse found.
        _actual_mode = "pure_win"
    elif strategy_mode in ("dutching", "hybrid") and has_dutch and n_win_tickets < 2:
        # (Audit R9-MINOR-1+2): Budget was too small to place both Dutch win bets.
        # Correct both resolved_mode and verdict_summary to reflect the single-bet reality.
        _actual_mode = "pure_win"
        win_suffix = f" (budget ¥{budget:,} only fits 1 bet — increase budget for full Dutch)"
        verdict_summary = f"My bet: No. {anchor.post_position} {anchor_name} to win{win_suffix}"
    elif strategy_mode == "hybrid" and has_dutch and n_win_tickets >= 2 and _n_exotic_allocated == 0 and not _exotics_budget_pruned:
        # (Audit R12-MINOR-1): Dutch executed correctly but no exotics qualify (not budget-pruned,
        # just no positive-EV exotics in this race).  The actual execution is a pure Dutch split;
        # report 'dutching' so STRATEGY_META accurately reflects the executed strategy.
        # Excludes the _exotics_budget_pruned case (R11-MINOR-1): there, exotics DID qualify
        # but were pruned by the budget cap — resolved_mode stays 'hybrid' to signal that
        # increasing the budget would restore the full hybrid portfolio.
        _actual_mode = "dutching"
    elif strategy_mode == "hybrid" and not has_dutch and _n_exotic_allocated == 0:
        # (Audit R12-MINOR-2): Single-anchor hybrid with no qualifying exotics — the actual
        # execution was a plain win bet.  Downgrade to 'pure_win' so STRATEGY_META is honest.
        _actual_mode = "pure_win"

    return StakingPlan(
        budget=budget,
        simple_bet=simple_bet,
        portfolio_tickets=allocated_tickets,
        anchor_horse=anchor,
        verdict_summary=verdict_summary,
        simple_bet_label=simple_bet_label,
        resolved_mode=_actual_mode,
    )


# ---------------------------------------------------------------------------
# 4. End-to-End Race Betting Analysis Helper
# ---------------------------------------------------------------------------

def analyze_race_betting(
    race_id: int,
    budget: int = 1000,
    strategy_mode: str = "hybrid",
    model_version: Optional[str] = None,
    session: Any = None,
) -> dict:
    """
    Given a race_id and optional budget and strategy_mode, fetches data from the database,
    calculates fair pricing, multi-finish probability model, and staking plan,
    and returns a structured dict conforming to Section 6 of docs/BETTING_ANALYSIS_LOGIC.md.
    """
    from scraper.db import get_session

    def _fetch(s):
        # Fetch race details
        race_row = s.execute(
            text("""
                SELECT r.id, r.race_name, r.race_name_jp, r.race_number, r.date,
                       c.name AS course_name, c.name_jp AS course_name_jp
                FROM races r
                LEFT JOIN courses c ON c.id = r.course_id
                WHERE r.id = :race_id
            """),
            {"race_id": race_id},
        ).fetchone()

        if not race_row:
            return None, [], []

        # Fetch entries
        entries = s.execute(
            text("""
                SELECT e.id, e.post_position, e.odds_win, e.popularity,
                       h.name AS horse_name, h.name_jp AS horse_name_jp
                FROM entries e
                LEFT JOIN horses h ON h.id = e.horse_id
                WHERE e.race_id = :race_id
                ORDER BY e.post_position
            """),
            {"race_id": race_id},
        ).fetchall()

        # Fetch latest predictions for this race
        q_pred = """
            SELECT p.entry_id, p.win_prob, p.edge, p.model_version
            FROM predictions p
            WHERE p.race_id = :race_id
        """
        params = {"race_id": race_id}
        if model_version:
            q_pred += " AND p.model_version = :mv"
            params["mv"] = model_version
        else:
            q_pred += """
              AND p.id IN (
                  SELECT id FROM (
                      SELECT id, ROW_NUMBER() OVER (
                          PARTITION BY entry_id ORDER BY created_at DESC, id DESC
                      ) AS rn
                      FROM predictions
                      WHERE race_id = :race_id
                  ) t WHERE rn = 1
              )
            """

        predictions = s.execute(text(q_pred), params).fetchall()
        return race_row, entries, predictions

    if session is not None:
        race_row, entry_rows, pred_rows = _fetch(session)
    else:
        with get_session() as s:
            race_row, entry_rows, pred_rows = _fetch(s)

    if not race_row:
        raise ValueError(f"Race {race_id} not found")

    pred_map = {p.entry_id: float(p.win_prob or 0.0) for p in pred_rows}

    # (Audit NEW-FLAW-3): Track ML prediction coverage so we can warn the user when
    # EV values are computed from market-implied probs (1/odds) rather than the model.
    # When a horse lacks an ML prediction, its win_prob = 1/mkt_odds, making
    # EV = (1/odds × odds) − 1 = 0.0 always — silent garbage masquerading as analysis.
    n_entries = len(entry_rows)
    n_with_pred = sum(
        1 for r in entry_rows
        if pred_map.get(r.id) is not None and pred_map.get(r.id, 0) > 0
    )
    ml_coverage = n_with_pred / max(1, n_entries)

    # If predictions are missing, fallback to market-implied probabilities
    entries_data = []
    for r in entry_rows:
        mkt_odds = float(r.odds_win) if r.odds_win is not None else None
        win_prob = pred_map.get(r.id)
        if win_prob is None:
            if mkt_odds is not None and mkt_odds > 1.0:
                win_prob = 1.0 / mkt_odds
            else:
                win_prob = 1.0 / max(1, n_entries)
        else:
            win_prob = max(0.0001, float(win_prob))

        entries_data.append({
            "post_position": r.post_position,
            "horse_name": r.horse_name,
            "horse_name_jp": r.horse_name_jp,
            "odds": mkt_odds,
            "win_prob": win_prob,
        })

    # Calculate pricing and staking plan
    pricing = calculate_pricing_breakdown(entries_data)
    staking = construct_staking_plan(pricing, budget=budget, strategy_mode=strategy_mode)

    # (Audit R4-INTEGRITY-3 + R5-LOGIC-4): Use the resolved_mode stored in the StakingPlan
    # instead of calling select_adaptive_strategy a second time.  The second call was
    # redundant (deterministic, same input) and left PASS races showing strategy_meta["auto"]
    # instead of the correct resolved outcome.
    _resolved_mode = staking.resolved_mode

    # Format full race name
    r_dict = dict(race_row._mapping)
    venue = r_dict.get("course_name") or r_dict.get("course_name_jp") or ""
    r_num = r_dict.get("race_number") or ""
    r_name = r_dict.get("race_name_jp") or r_dict.get("race_name") or "Race"
    full_race_name = f"{venue} {r_num}R — {r_name}".strip() if venue and r_num else r_name

    anchor_name = staking.anchor_horse.horse_name_jp or staking.anchor_horse.horse_name
    has_mkt_odds = any(h.market_odds is not None and h.market_odds > 1.0 for h in pricing)
    pos_str = f"No. {staking.anchor_horse.post_position} " if staking.anchor_horse.post_position > 0 else ""

    if not has_mkt_odds:
        recommended_horse_str = f"{pos_str}{anchor_name}"
        action_str = f"Fair: {staking.anchor_horse.fair_odds}x ({staking.anchor_horse.win_prob_pct}% win chance)"
    elif not staking.portfolio_tickets:
        recommended_horse_str = "PASS — No Value Bets"
        action_str = "Pass race"
    else:
        recommended_horse_str = f"{pos_str}{anchor_name}"
        action_str = "to win"

    # INTEGRITY NOTE (Bug #13): The stats below (historical_roi, hit_rate, sharpe) are
    # STATIC TARGET STRINGS and are NOT computed from live backtests. They represent
    # theoretical design targets from early simulation work and must be manually updated
    # whenever strategy logic changes significantly. The 'stats_validated' flag is
    # explicitly set to False so the frontend can display a disclaimer.
    # (Audit INTEGRITY-1: these numbers were served to users as if real.)
    #
    # (Audit R7-SIG-4): Removed the "auto" and "adaptive" entries — resolved_mode can
    # never be "auto" or "adaptive" (the auto path resolves to the actual chosen mode
    # before storing resolved_mode).  These entries were dead code and showed misleading
    # '+88.5% ROI' stats if ever accidentally reached via the fallback chain.
    STRATEGY_META = {
        "hybrid": {
            "id": "hybrid",
            "name": "Balanced Portfolio",
            "icon": "🎯",
            "tagline": "75% Core Win + 25% Elite Exotics",
            "historical_roi": "+57.5% – +77.3%",
            "hit_rate": "37.7%",
            "sharpe": 3.02,
            "stats_validated": False,
        },
        "pure_win": {
            "id": "pure_win",
            "name": "Pure Value Win",
            "icon": "⚡",
            "tagline": "100% Single Top Value Pick",
            "historical_roi": "+81.6% – +109.2%",
            "hit_rate": "33.5%",
            "sharpe": 3.49,
            "stats_validated": False,
        },
        "dutching": {
            "id": "dutching",
            "name": "Dual Dutching",
            "icon": "🛡️",
            "tagline": "Dual Value Win Split — Lowest Drawdown (8.7%)",
            "historical_roi": "+84.3%",
            "hit_rate": "37.0%",
            "sharpe": 3.81,
            "stats_validated": False,
        },
        # (Audit R6-META-1): Explicit entries for non-betting outcomes so that
        # STRATEGY_META.get(_resolved_mode) returns sensible meta instead of
        # falling back to the 'auto' entry and showing misleading historical stats
        # (e.g. '+88.5% ROI, Sharpe 3.95' on a PASS race).
        "pass": {
            "id": "pass",
            "name": "Pass (No Bet)",
            "icon": "⏭️",
            "tagline": "No mathematical edge found — capital preserved",
            "historical_roi": "N/A",
            "hit_rate": "N/A",
            "sharpe": 0,
            "stats_validated": True,
        },
        "pending": {
            "id": "pending",
            "name": "Awaiting Market Odds",
            "icon": "⏳",
            "tagline": "Official JRA betting not yet open — fair odds only",
            "historical_roi": "N/A",
            "hit_rate": "N/A",
            "sharpe": 0,
            "stats_validated": True,
        },
    }

    data_quality = {
        "ml_coverage": round(ml_coverage, 3),
        "horses_with_predictions": n_with_pred,
        "total_horses": n_entries,
        # Warn when <70% of the field has ML predictions; EV values for the remaining
        # horses are computed from market-implied probabilities (EV = 0 by definition)
        # and should not be relied upon.
        "coverage_warning": ml_coverage < 0.7,
        "coverage_warning_msg": (
            f"⚠️ ML predictions available for only {n_with_pred}/{n_entries} runners "
            f"({ml_coverage:.0%}). EV for uncovered horses is market-implied (always 0) "
            "and may not reflect true mispricing."
        ) if ml_coverage < 0.7 else None,
    }

    return {
        "race_id": race_id,
        "race_name": full_race_name,
        "budget": budget,
        "strategy_mode": strategy_mode,
        "strategy_meta": STRATEGY_META.get(_resolved_mode, STRATEGY_META.get(strategy_mode, STRATEGY_META["hybrid"])),
        "data_quality": data_quality,
        "headline": {
            "recommended_horse": recommended_horse_str,
            "post_position": staking.anchor_horse.post_position,
            "action": action_str,
            "summary": staking.verdict_summary,
        },
        # (Audit R7-MINOR-4): Use BetTicket.to_dict() instead of a manual projection
        # so type_en, selection_display, weight_pct are included, matching the field
        # set returned by StakingPlan.to_dict() callers.
        "staking_plan": {
            "budget": budget,
            "tickets": [t.to_dict() for t in staking.portfolio_tickets],
            "simple_bet_label": staking.simple_bet_label,
        },
        # (Audit R7-SIG-3): Use HorsePricing.to_dict() instead of a manual projection
        # so ev, is_top_value, is_favorite, is_market_favorite are included.
        # The old manual projection returned a stripped table with no per-horse EV,
        # forcing the frontend to work without EV data from this endpoint.
        "pricing_table": [h.to_dict() for h in pricing],
    }
