"""
Shift-Aware Episode Diagnostic
==============================
ONE QUESTION: how many shift-aware TRAINING episodes actually carry usable signal?
Tests whether "shift-aware training barely helps" is a real finding or an artifact
of dead training signal.

Shift-aware episodes draw support and query each from a SINGLE Murcko scaffold group
(data.EpisodeSampler._get_episode_indices_shift_aware). A single scaffold series is
often all-active or all-inactive, in which case:
  * support single-class  -> the classification head masks the episode out
                             (valid_mask=False) -> ZERO gradient. The episode is wasted.
  * query single-class     -> no delta-AUPRC signal for that episode.

If a large fraction of shift-aware episodes are wasted, then "shift-aware == random"
says more about wasted episodes than about OOD training being useless.

This samples episodes the exact way training does (streaming from disk, same sampler)
WITHOUT training, and reports the wasted fraction. No GPU, no checkpoint needed.

HOW TO RUN: edit the CONFIG block at the bottom of this file, then:
    python Analysis/data/shift_aware_episodes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FSMOL_TRAIN  # noqa: E402
from data import _load_assay_file, EpisodeSampler, build_fair_split_indices  # noqa: E402


def get_files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl.gz"))


def diagnose(n_episodes, n_support, n_query, seed):
    """Measure the CURRENT shift-aware sampler by replicating its exact call to
    build_fair_split_indices (so we can observe whether it falls back to random)."""
    files = get_files(FSMOL_TRAIN)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)  # build_fair_split_indices uses np.random (passed as rng)
    sampler = EpisodeSampler(n_support, n_query)

    stats = {"sampled": 0, "fell_back_to_random": 0, "scaffold_episode": 0,
             "support_single_class": 0, "query_single_class": 0, "usable": 0}
    sup_sizes, sup_active_frac = [], []

    attempts = 0
    while stats["sampled"] < n_episodes and attempts < n_episodes * 50:
        attempts += 1
        path = files[rng.randint(len(files))]
        ds = _load_assay_file(path)
        if len(ds) < sampler.n_support_min or ds.binary_labels is None:
            continue
        stats["sampled"] += 1

        n_total = len(ds)
        adaptive_target = min(n_support, max(2, n_total // 2))
        sp = build_fair_split_indices(
            n_total, ds.scaffold_groups, ds.binary_labels,
            adaptive_target, "scaffold", np.random,
            n_query_cap=n_query, require_both_classes=True,
            min_support=min(8, adaptive_target))
        if sp is None:
            stats["fell_back_to_random"] += 1
            continue
        stats["scaffold_episode"] += 1
        s_idx, q_idx = sp
        ys, yq = ds.binary_labels[s_idx], ds.binary_labels[q_idx]
        sup_sizes.append(len(s_idx))
        sup_active_frac.append(float((ys == 1).mean()))
        sup_one = len(np.unique(ys)) < 2
        qry_one = len(np.unique(yq)) < 2
        if sup_one:
            stats["support_single_class"] += 1
        if qry_one:
            stats["query_single_class"] += 1
        if not sup_one and not qry_one:
            stats["usable"] += 1

    return stats, np.array(sup_active_frac), np.array(sup_sizes)


def report(stats, sup_frac, sup_sizes, n_support):
    N  = max(stats["sampled"], 1)
    sc = max(stats["scaffold_episode"], 1)  # genuine scaffold-disjoint episodes

    print("\n" + "=" * 64)
    print(f"SHIFT-AWARE EPISODE DIAGNOSTIC  (n_support={n_support})")
    print("=" * 64)
    print(f"  Episodes sampled                 : {stats['sampled']:>7,}")
    print(f"  Fell back to random              : {stats['fell_back_to_random']:>7,}  "
          f"({100*stats['fell_back_to_random']/N:.1f}% of all)")
    print(f"  Genuine scaffold-disjoint episode: {stats['scaffold_episode']:>7,}  "
          f"({100*stats['scaffold_episode']/N:.1f}% of all)")
    print(f"  --- of the {sc:,} scaffold episodes ---")
    print(f"  WASTED: support single-class     : {stats['support_single_class']:>7,}  "
          f"({100*stats['support_single_class']/sc:.1f}%)  <- should be ~0 now")
    print(f"  Query single-class               : {stats['query_single_class']:>7,}  "
          f"({100*stats['query_single_class']/sc:.1f}%)  (ok for BCE training)")
    print(f"  Usable (both sides 2 classes)    : {stats['usable']:>7,}  "
          f"({100*stats['usable']/sc:.1f}%)")
    if len(sup_sizes):
        print(f"  Support size                     : mean={sup_sizes.mean():.1f}  "
              f"min={sup_sizes.min()}  max={sup_sizes.max()}")
    if len(sup_frac):
        print(f"  Support active-fraction          : mean={sup_frac.mean():.3f}  "
              f"std={sup_frac.std():.3f}  "
              f"[%==0: {100*(sup_frac==0).mean():.1f}, %==1: {100*(sup_frac==1).mean():.1f}]")
    print("=" * 64)
    print("Target: low fallback %, support single-class ~0%. Then shift-aware training\n"
          "actually fires and a shift-aware-vs-random comparison becomes interpretable.")


if __name__ == "__main__":
    # ===== CONFIG - edit these, then run (no arguments) =====
    N_EPISODES = 2000   # episodes to sample (more = tighter percentages, slower)
    N_SUPPORT  = 64     # support size to diagnose - match the training config in main.py
    N_QUERY    = 256
    SEED       = 42
    # =======================================================
    stats, sup_frac, sup_sizes = diagnose(N_EPISODES, N_SUPPORT, N_QUERY, SEED)
    report(stats, sup_frac, sup_sizes, N_SUPPORT)
