#!/usr/bin/env python3
"""Floors: the lowest mean guesses any strategy can reach on the development games, pool by pool.

Each answer pool (never-used list words, reused answers, unlisted new words) is taken as uniform, and the
floor gives the solver the pool in advance, so it also bounds a solver that does not know it.
Reads data/seed, data/raw/nyt_daily_answers.jsonl through the development cutoff only, and the redacted
public rolling evidence. Writes nothing except the CSV named on the `never` command line.
Needs Python 3.12 with numpy and numba. Runs on 4 threads unless NUMBA_NUM_THREADS is set.
Results and method: docs/superpowers/plans/2026-10-03-subthree-research.md, "October 27 floor analysis".

  check            floor <= exact optimum <= strategy on small pools; replay of the 2026-08-25 pool
  pool DATE        floor and a concrete strategy for the never-used pool on DATE
  never CSV        floor on every third development date and oracle play of the day's answer (slow)
  summary CSV      totals, per-fold floors and the matched comparison from a `never` CSV
  projection       floors for random smaller never-used pools
  reused           floors for the 17 reuse days under two reuse pools
  dormant          proves the new-word floor is at least T for increasing T (slow)
  mix              per-fold answer mix and the chance that a 30-game window averages <= 3.0
  signals          information in recent-answer features; shape of the new words
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEV_CUTOFF = "2026-08-26"  # never read the reserved seal or anything after development
GREEN = 242


def read_words(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if len(w) == 5 and w.isalpha() and w.isascii():
            out.append(w)
    return out


def load():
    guesses = read_words(REPO / "data/seed/valid_guesses.txt")
    seed = read_words(REPO / "data/seed/candidate_answers.txt")
    hist = []
    for line in (REPO / "data/raw/nyt_daily_answers.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["print_date"] <= DEV_CUTOFF:
            hist.append((r["print_date"], r["solution"].lower(), r["id"]))
    hist.sort()
    ev = json.loads((REPO / "docs/evidence/september-audit-rolling-public-v1.json").read_text(encoding="utf-8"))
    games = [(g["outcome"]["date"], g["outcome"]["status"], g["outcome"]["guesses"]) for g in ev["baseline"]["games"]]
    return guesses, seed, hist, games


def classify(seed, hist, date):
    """Pools on `date`: never-used seed words, previously used answers, and the answer's own class."""
    used_before = {}
    for d, w, _ in hist:
        if d < date:
            used_before.setdefault(w, d)
    never = sorted(set(seed) - set(used_before))
    answer = next((w for d, w, _ in hist if d == date), None)
    if answer in used_before:
        cls = "reused"
    elif answer in set(seed):
        cls = "never"
    else:
        cls = "new"
    return never, used_before, answer, cls


def score(status, guesses):
    return guesses if status == "solved" else 7


def encode(words):
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)


@njit
def _pattern(g, a):
    counts = np.zeros(26, np.int8)
    green = np.zeros(5, np.bool_)
    for i in range(5):
        if g[i] == a[i]:
            green[i] = True
        else:
            counts[a[i]] += 1
    res = 0
    mul = 1
    for i in range(5):
        v = 0
        if green[i]:
            v = 2
        elif counts[g[i]] > 0:
            v = 1
            counts[g[i]] -= 1
        res += v * mul
        mul *= 3
    return res


@njit(parallel=True)
def pattern_rows(answers_enc, guesses_enc):
    """PT[a, g]: feedback code 0..242 (242 = all green) for answer row a and guess g."""
    A = answers_enc.shape[0]
    G = guesses_enc.shape[0]
    out = np.empty((A, G), np.uint8)
    for a in prange(A):
        for g in range(G):
            out[a, g] = _pattern(guesses_enc[g], answers_enc[a])
    return out


@njit
def _distinct(PT, rows, k, g2, bits):
    bits[0] = 0
    bits[1] = 0
    bits[2] = 0
    bits[3] = 0
    for j in range(k):
        p = PT[rows[j], g2]
        bits[p >> 6] |= np.uint64(1) << np.uint64(p & 63)
    c = 0
    for w in range(4):
        x = bits[w]
        while x:
            x &= x - np.uint64(1)
            c += 1
    return c


@njit
def bucket_lb(PT, rows, k, row_guess, bits, all_guesses):
    """Lower bound on the summed guesses for a bucket of k equally likely answers (rows into PT).

    Every next guess g2 splits the bucket into sub-buckets; each sub-bucket b' costs at least 2|b'|-1,
    so the bucket costs at least 3k - max_g2(parts(g2) + [g2 in bucket]).
    """
    if k == 1:
        return 1
    if k == 2:
        return 3
    best = 0
    # members first: a member that splits the rest completely reaches the floor 2k-1
    for j in range(k):
        g2 = row_guess[rows[j]]
        if g2 < 0:
            continue
        v = _distinct(PT, rows, k, g2, bits) + 1
        if v > best:
            best = v
            if best == k + 1:
                return 2 * k - 1
    if best >= k:  # a non-member gives at most k
        return 3 * k - best
    for g2 in range(all_guesses):
        v = _distinct(PT, rows, k, g2, bits)
        if v > best:
            best = v
            if best == k:
                break
    return 3 * k - best


@njit(parallel=True)
def root_lb2(PT, pool_rows, row_guess, cand, threshold):
    """For each first guess in `cand`: n + sum over non-green buckets of bucket_lb.

    Stops early (returns a value >= threshold) once a first guess cannot beat `threshold`.
    """
    n = pool_rows.shape[0]
    G = PT.shape[1]
    out = np.empty(cand.shape[0], np.int64)
    for ci in prange(cand.shape[0]):
        g = cand[ci]
        bits = np.zeros(4, np.uint64)
        cnt = np.zeros(243, np.int64)
        for j in range(n):
            cnt[PT[pool_rows[j], g]] += 1
        start = np.zeros(244, np.int64)
        for p in range(243):
            start[p + 1] = start[p] + cnt[p]
        fill = start[:243].copy()
        order = np.empty(n, np.int64)
        for j in range(n):
            p = PT[pool_rows[j], g]
            order[fill[p]] = pool_rows[j]
            fill[p] += 1
        # cheap part first: every non-green bucket costs at least 2k-1
        total = n
        for p in range(243):
            if p != GREEN and cnt[p] > 0:
                total += 2 * cnt[p] - 1
        if total >= threshold:
            out[ci] = total
            continue
        for p in range(243):
            k = cnt[p]
            if p == GREEN or k < 3:
                continue
            lb = bucket_lb(PT, order[start[p]:start[p] + k], k, row_guess, bits, G)
            total += lb - (2 * k - 1)
            if total >= threshold:
                break
        out[ci] = total
    return out


def build(guesses, answers):
    """Pattern rows for `answers` and, per answer row, its guess index (-1 if not a valid guess)."""
    PT = pattern_rows(encode(answers), encode(guesses))
    gidx = {w: i for i, w in enumerate(guesses)}
    row_guess = np.array([gidx.get(w, -1) for w in answers], dtype=np.int64)
    return PT, row_guess


def build_answers(guesses, seed, hist):
    answers = sorted(set(seed) | {w for _, w, _ in hist})
    PT, row_guess = build(guesses, answers)
    return answers, {w: i for i, w in enumerate(answers)}, PT, row_guess


def floor_for_pool(PT, row_guess, rows, top=400):
    """Floor (summed guesses) over every first guess, plus the best first guesses by floor."""
    rows = np.asarray(rows, dtype=np.int64)
    sub = PT[rows]
    G = sub.shape[1]
    idx = sub.astype(np.int64) + 243 * np.arange(G)[None, :]
    parts = (np.bincount(idx.ravel(), minlength=243 * G).reshape(G, 243) > 0).sum(1)
    order = np.argsort(-parts, kind="stable").astype(np.int64)
    T = int(root_lb2(PT, rows, row_guess, order[:top], np.int64(1 << 60)).min())
    allv = root_lb2(PT, rows, row_guess, order, np.int64(T + 1))
    ranked = order[np.argsort(allv, kind="stable")]
    return int(allv.min()), ranked[:12], np.sort(allv)[:12]


class Strategy:
    """A concrete strategy: at each node try the `width` best guesses by the bucket bound and recurse."""

    def __init__(self, PT, row_guess, width=4):
        self.PT, self.row_guess, self.width = PT, row_guess, width
        self.G = PT.shape[1]
        self.memo = {}
        self.choice = {}

    def cost(self, rows):
        rows = tuple(sorted(rows))
        k = len(rows)
        if k <= 2:
            self.choice[rows] = int(self.row_guess[rows[0]])
            return 1 if k == 1 else 3
        if rows in self.memo:
            return self.memo[rows]
        sub = self.PT[list(rows)]  # (k, G)
        srt = np.sort(sub, axis=0)
        c = 1 + (np.diff(srt, axis=0) != 0).sum(0)
        member = np.zeros(self.G, np.int64)
        for r in rows:
            if self.row_guess[r] >= 0:
                member[self.row_guess[r]] = 1
        score_ = c + member  # bucket_lb = 3k - score_
        best = None
        for g2 in np.argsort(-score_, kind="stable")[: self.width]:
            if best is not None and 3 * k - score_[g2] >= best:
                break
            pats = sub[:, g2]
            tot = k
            for p in np.unique(pats):
                if p == GREEN:
                    continue
                tot += self.cost([rows[i] for i in np.nonzero(pats == p)[0]])
                if best is not None and tot >= best:
                    break
            if best is None or tot < best:
                best, arg = tot, int(g2)
        self.memo[rows] = best
        self.choice[rows] = arg
        return best

    def root(self, rows, g):
        rows = np.asarray(rows)
        pats = self.PT[rows, g]
        tot = len(rows)
        for p in np.unique(pats):
            if p != GREEN:
                tot += self.cost(list(rows[pats == p]))
        return tot

    def play(self, rows, g, answer_row):
        """Guesses this strategy needs for one answer, opening with guess g (after root())."""
        cand = np.asarray(rows)
        n = 1
        while self.row_guess[answer_row] != g:
            cand = cand[self.PT[cand, g] == self.PT[answer_row, g]]
            g = int(self.row_guess[cand[0]]) if len(cand) == 1 else self.choice[tuple(sorted(int(r) for r in cand))]
            n += 1
        return n


def cmd_check(_args):
    def fb(g, a):
        code = int(_pattern(encode([g])[0], encode([a])[0]))
        return "".join(str((code // 3**i) % 3) for i in range(5))

    cases = {("array", "royal"): "01021", ("speed", "abide"): "00101", ("speed", "erase"): "10110",
             ("eerie", "elder"): "21100", ("salet", "cigar"): "01000", ("geese", "eerie"): "02102",
             ("lever", "eerie"): "02011", ("cigar", "cigar"): "22222"}
    for (g, a), want in cases.items():
        assert fb(g, a) == want, (g, a, fb(g, a), want)
    print("feedback cases ok")

    guesses, seed, hist, _ = load()
    answers, arow, PT, row_guess = build_answers(guesses, seed, hist)

    def exact(rows, memo):
        """True optimum of the summed guesses over all guesses, with bound pruning (rows: sorted tuple)."""
        k = len(rows)
        if k <= 2:
            return 1 if k == 1 else 3
        if rows in memo:
            return memo[rows]
        sub = PT[list(rows)]
        srt = np.sort(sub, axis=0)
        c = 1 + (np.diff(srt, axis=0) != 0).sum(0)
        member = np.zeros(PT.shape[1], np.int64)
        for r in rows:
            member[row_guess[r]] = 1
        lb = 3 * k - (c + member)
        best = None
        for g in np.argsort(lb, kind="stable"):
            if best is not None and lb[g] >= best:
                break
            pats = sub[:, g]
            tot = k
            for p in np.unique(pats):
                if p != GREEN:
                    tot += exact(tuple(rows[i] for i in np.nonzero(pats == p)[0]), memo)
            if best is None or tot < best:
                best = tot
        memo[rows] = best
        return best

    rng = random.Random(7)
    for size in (12, 25, 40, 60):
        for _ in range(3):
            rows = tuple(sorted(rng.sample(range(len(answers)), size)))
            fl, firsts, _ = floor_for_pool(PT, row_guess, list(rows), top=50)
            ex = exact(rows, {})
            strat = Strategy(PT, row_guess)
            ach = min(strat.root(list(rows), int(g)) for g in firsts[:3])
            assert fl <= ex <= ach, (size, fl, ex, ach)
            print(f"pool {size:3d}: floor {fl} <= exact {ex} <= strategy {ach}")

    # replay every answer with the plain feedback function, independent of the pattern table
    never = classify(seed, hist, "2026-08-25")[0]
    rows = [arow[w] for w in never]
    fl, firsts, _ = floor_for_pool(PT, row_guess, rows)
    strat = Strategy(PT, row_guess)
    g0 = int(firsts[0])
    cost = strat.root(rows, g0)
    total = worst = 0
    for ans in never:
        cand, g, n = list(never), guesses[g0], 1
        while g != ans:
            f = fb(g, ans)
            cand = [w for w in cand if fb(g, w) == f]
            g = cand[0] if len(cand) == 1 else guesses[strat.choice[tuple(sorted(arow[w] for w in cand))]]
            n += 1
        total += n
        worst = max(worst, n)
    assert total == cost, (total, cost)
    print(f"replay 2026-08-25 pool {len(never)}: opener {guesses[g0]}, total {total} = strategy cost, "
          f"mean {total/len(never):.4f}, worst {worst}, floor {fl} ({fl/len(never):.4f})")


def cmd_pool(args):
    guesses, seed, hist, _ = load()
    answers, arow, PT, row_guess = build_answers(guesses, seed, hist)
    never = classify(seed, hist, args.date)[0]
    rows = [arow[w] for w in never]
    t = time.time()
    best, firsts, vals = floor_for_pool(PT, row_guess, rows)
    strat = Strategy(PT, row_guess)
    g0 = int(firsts[0])
    ach = strat.root(rows, g0)
    replayed = sum(strat.play(rows, g0, r) for r in rows)
    n = len(rows)
    print(f"{args.date} never-used pool {n}: floor {best} ({best/n:.4f}/game), strategy {ach} ({ach/n:.4f}/game, "
          f"replayed {replayed}), best openers by floor {[guesses[g] for g in firsts[:6]]} {list(map(int, vals[:6]))}"
          f" [{time.time()-t:.0f}s]")


def cmd_never(args):
    guesses, seed, hist, games = load()
    answers, arow, PT, row_guess = build_answers(guesses, seed, hist)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "cls", "selected_guesses", "pool", "floor_total", "floor_mean", "achievable_mean",
                    "best_first", "oracle_guesses_on_answer"])
        for i, (date, status, n) in enumerate(games):
            if i % 3 and i != len(games) - 1:
                continue
            t = time.time()
            never, _, answer, cls = classify(seed, hist, date)
            rows = [arow[x] for x in never]
            best, firsts, _ = floor_for_pool(PT, row_guess, rows)
            first = int(firsts[0])
            strat = Strategy(PT, row_guess)
            ach = strat.root(rows, first)
            oracle = strat.play(rows, first, arow[answer]) if cls == "never" else ""
            w.writerow([date, cls, score(status, n), len(rows), best, best / len(rows), ach / len(rows),
                        guesses[first], oracle])
            fh.flush()
            print(date, cls, len(rows), round(best / len(rows), 4), round(ach / len(rows), 4),
                  f"{time.time()-t:.0f}s", flush=True)
    summarize(args.csv, guesses, seed, hist, games)


def cmd_summary(args):
    guesses, seed, hist, games = load()
    summarize(args.csv, guesses, seed, hist, games)


def summarize(path, guesses, seed, hist, games):
    samp = list(csv.DictReader(open(path)))
    idx = {g[0]: i for i, g in enumerate(games)}
    xs = np.array([idx[r["date"]] for r in samp])
    fl = np.array([float(r["floor_mean"]) for r in samp])
    ach = np.array([float(r["achievable_mean"]) for r in samp])
    print(f"sampled dates {len(samp)}, largest strategy-minus-floor gap {float((ach - fl).max()):.4f}/game")
    tot_floor = tot_sel = 0.0
    folds = collections.defaultdict(list)
    for i, (date, status, n) in enumerate(games):
        if classify(seed, hist, date)[3] != "never":
            continue
        f = float(np.interp(i, xs, fl))
        tot_floor += f
        tot_sel += score(status, n)
        folds[i // 30].append((f, score(status, n)))
    days = sum(len(v) for v in folds.values())
    print(f"never-used days {days}: selected {tot_sel:.0f} ({tot_sel/days:.4f}), floor {tot_floor:.1f} "
          f"({tot_floor/days:.4f}), gap {tot_sel - tot_floor:.1f}")
    for k, v in sorted(folds.items()):
        print(f"  fold {k:2d}: {len(v):2d} days, floor {np.mean([x[0] for x in v]):.4f}, "
              f"selected {np.mean([x[1] for x in v]):.4f}")
    pairs = np.array([(int(r["oracle_guesses_on_answer"]), int(r["selected_guesses"])) for r in samp
                      if r["cls"] == "never"])
    d = pairs[:, 1] - pairs[:, 0]
    rng = np.random.default_rng(1)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"same {len(d)} never-used answers: oracle strategy {pairs[:, 0].mean():.4f}, selected "
          f"{pairs[:, 1].mean():.4f}, selected minus oracle {d.mean():+.4f} "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")


def cmd_projection(_args):
    guesses, seed, hist, _ = load()
    answers, arow, PT, row_guess = build_answers(guesses, seed, hist)
    never = classify(seed, hist, DEV_CUTOFF)[0]
    print(f"never-used pool on {DEV_CUTOFF}: {len(never)}", flush=True)
    rng = random.Random(11)
    for size in (450, 400, 350, 300, 250, 200, 150, 100, 50):
        res = []
        for _ in range(3):
            rows = [arow[w] for w in rng.sample(never, size)]
            fl, firsts, _ = floor_for_pool(PT, row_guess, rows)
            strat = Strategy(PT, row_guess)
            res.append((fl / size, min(strat.root(rows, int(g)) for g in firsts[:2]) / size))
        a = np.array(res)
        print(f"size {size:3d}: floor {a[:, 0].mean():.4f} (range {a[:, 0].min():.4f}-{a[:, 0].max():.4f}), "
              f"strategy {a[:, 1].mean():.4f}", flush=True)


def cmd_reused(_args):
    guesses, seed, hist, games = load()
    answers, arow, PT, row_guess = build_answers(guesses, seed, hist)
    first = {}
    for d, w, i in hist:
        first.setdefault(w, d)
    early = {w for w, d in first.items() if d < "2022-01-05"}  # puzzles 1..200
    tot_a = tot_b = 0.0
    for date, status, n in games:
        never, used, answer, cls = classify(seed, hist, date)
        if cls != "reused":
            continue
        reused_already = {w for d, w, _ in hist if d < date and first[w] < d}
        age = {w: (dt.date.fromisoformat(date) - dt.date.fromisoformat(d0)).days for w, d0 in used.items()}
        pool_a = sorted(w for w in used if age[w] >= 1400 and w not in reused_already)
        pool_b = sorted(w for w in early if w in used and w not in reused_already)
        fa = floor_for_pool(PT, row_guess, [arow[w] for w in pool_a])[0] / len(pool_a)
        fb = floor_for_pool(PT, row_guess, [arow[w] for w in pool_b])[0] / len(pool_b)
        tot_a += fa
        tot_b += fb
        print(f"{date} answer age {age[answer]:5d} d, in (a) {answer in pool_a}, in (b) {answer in pool_b}; "
              f"(a) {len(pool_a)} words floor {fa:.4f}, (b) {len(pool_b)} words floor {fb:.4f}; "
              f"selected {score(status, n)}", flush=True)
    print(f"sum over reuse days: (a) answers >= 1400 days old {tot_a:.1f}, (b) answers of puzzles 1-200 {tot_b:.1f}")


def cmd_dormant(_args):
    guesses, seed, hist, games = load()
    used = classify(seed, hist, DEV_CUTOFF)[1]
    dormant = sorted(set(guesses) - set(seed) - set(used))
    news = [classify(seed, hist, d)[2] for d, _, _ in games if classify(seed, hist, d)[3] == "new"]
    print(f"unlisted pool on {DEV_CUTOFF}: {len(dormant)} words; the {len(news)} development new words are all "
          f"valid guesses: {set(news) <= set(guesses)}", flush=True)
    PT, row_guess = build(guesses, dormant)
    rows = np.arange(len(dormant), dtype=np.int64)
    cand = np.arange(len(guesses), dtype=np.int64)
    N = len(dormant)
    for T in (3.2, 3.4, 3.6, 3.8):
        t = time.time()
        out = root_lb2(PT, rows, row_guess, cand, np.int64(int(T * N)))
        ok = out.min() >= int(T * N)
        print(f"T={T}: floor >= {T} proven: {ok} [{time.time()-t:.0f}s]", flush=True)
        if not ok:
            break


def p_window_at_most(dist, games=30, target=3.0):
    """Exact P(mean of `games` iid draws from dist <= target); dist: {guesses: prob}."""
    pmf = np.zeros(7 * games + 1)
    pmf[0] = 1.0
    for _ in range(games):
        new = np.zeros_like(pmf)
        for g, p in dist.items():
            new[g:] += p * pmf[: len(pmf) - g]
        pmf = new
    return pmf[: int(target * games) + 1].sum()


def cmd_mix(_args):
    guesses, seed, hist, games = load()
    rows = []
    for date, status, n in games:
        never, _, _, cls = classify(seed, hist, date)
        rows.append((date, cls, score(status, n), len(never)))
    for cls in ("never", "reused", "new"):
        v = [r[2] for r in rows if r[1] == cls]
        print(f"{cls:6s} {len(v):3d} games, {sum(v):4d} guesses, mean {sum(v)/len(v):.4f}, "
              f"{dict(sorted(collections.Counter(v).items()))}")
    print("fold start  never reused new | mean never reused    new |  all  | never-used pool")
    for f in range(len(rows) // 30):
        chunk = rows[30 * f: 30 * f + 30]
        c = collections.Counter(r[1] for r in chunk)
        m = {k: np.mean([r[2] for r in chunk if r[1] == k]) if c[k] else float("nan") for k in ("never", "reused", "new")}
        print(f"{chunk[0][0]}  {c['never']:5d} {c['reused']:6d} {c['new']:3d} | {m['never']:10.3f} {m['reused']:6.3f}"
              f" {m['new']:6.3f} | {np.mean([r[2] for r in chunk]):.3f} | {chunk[0][3]}->{chunk[-1][3]}")
    scores = [r[2] for r in rows]
    dev = collections.Counter(scores)
    print(f"P(30-game mean <= 3.0) with the development distribution: "
          f"{p_window_at_most({g: c / len(scores) for g, c in dev.items()}):.3f}")
    sealed = json.loads((REPO / "benchmarks/predictive/sealed-selected-v20-20260726.json").read_text(encoding="utf-8"))

    def find(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    yield v
                yield from find(v, key)
        elif isinstance(o, list):
            for v in o:
                yield from find(v, key)

    sc = next(find(sealed, "solved_in_guess_counts"))
    counts = {i + 1: c for i, c in enumerate(sc) if c}
    print(f"P(30-game mean <= 3.0) with the sealed test's distribution: "
          f"{p_window_at_most({g: c / sum(counts.values()) for g, c in counts.items()}):.3f}")
    last = collections.Counter(scores[-30:])
    print(f"P(30-game mean <= 3.0) with the last fold's distribution: "
          f"{p_window_at_most({g: c / 30 for g, c in last.items()}):.3f}")


def cmd_signals(_args):
    guesses, seed, hist, games = load()
    answer_on = {d: w for d, w, _ in hist}
    dates_all = sorted(answer_on)

    def feats(w, prev, pos):
        p1, last7 = prev[-1], prev[-7:]
        return {
            "shared letters with yesterday": len(set(w) & set(p1)),
            "max shared letters, last 7 days": max(len(set(w) & set(p)) for p in last7),
            "same first letter as yesterday": w[0] == p1[0],
            "first letter seen in last 7 days": any(w[0] == p[0] for p in last7),
            "same last two letters as a last-30 answer": any(w[-2:] == p[-2:] for p in prev[-30:]),
            "first letter": w[0],
            "alphabetical decile in pool": pos[w],
            "has a repeated letter": len(set(w)) < 5,
            "vowel count": sum(c in "aeiou" for c in w),
        }

    rng = random.Random(0)
    obs = collections.defaultdict(collections.Counter)
    null = collections.defaultdict(collections.Counter)
    share = collections.defaultdict(collections.Counter)
    days = 0
    news, dorm = [], []
    for date, _, _ in games:
        never, used, answer, cls = classify(seed, hist, date)
        if cls == "new":
            news.append(answer)
            dorm = sorted(set(guesses) - set(seed) - set(used))
        if cls != "never":
            continue
        days += 1
        i = dates_all.index(date)
        prev = [answer_on[d] for d in dates_all[max(0, i - 30):i]]
        pos = {w: k * 10 // len(never) for k, w in enumerate(never)}
        for w in never:
            for k, v in feats(w, prev, pos).items():
                share[k][v] += 1 / len(never)
        for target, store in ((answer, obs), (rng.choice(never), null)):
            for k, v in feats(target, prev, pos).items():
                store[k][v] += 1

    def kl(counter, sh):
        tot = sum(counter.values())
        return sum((c / tot) * math.log((c / tot) / (sh[v] / days)) for v, c in counter.items() if c)

    print(f"{days} never-used development days; KL(answers || pool) in nats, and the same for a random pool word")
    for k in obs:
        print(f"  {k:45s} {kl(obs[k], share[k]):7.4f}   shuffled {kl(null[k], share[k]):7.4f}")
    print(f"{len(news)} new-word days against the {len(dorm)}-word unlisted pool of the last one:")
    tests = {"ends in s": lambda w: w.endswith("s"), "ends in ed": lambda w: w.endswith("ed"),
             "has j/q/x/z": lambda w: any(c in "jqxz" for c in w), "repeated letter": lambda w: len(set(w)) < 5}
    for name, fn in tests.items():
        print(f"  {name:16s} new words {sum(map(fn, news)):2d}/{len(news)}, unlisted pool {sum(map(fn, dorm))/len(dorm):6.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "projection", "reused", "dormant", "mix", "signals"):
        sub.add_parser(name)
    sub.add_parser("pool").add_argument("date")
    sub.add_parser("never").add_argument("csv")
    sub.add_parser("summary").add_argument("csv")
    args = parser.parse_args(argv)
    {"check": cmd_check, "pool": cmd_pool, "never": cmd_never, "summary": cmd_summary,
     "projection": cmd_projection, "reused": cmd_reused, "dormant": cmd_dormant, "mix": cmd_mix,
     "signals": cmd_signals}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
