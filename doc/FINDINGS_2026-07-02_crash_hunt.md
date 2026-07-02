# Crash hunt — the combo std::out_of_range is STILL OPEN (2026-07-02)

Follow-up to FINDINGS_2026-06-26 Finding 2 (the deployment blocker) and the
be78056 fix + VERIFICATION_2026-06-27 audit. Run on this branch's HEAD, combo
config (X-ICP ternary fullRatio 0.05 + governor 0.15, LiDAR-only) on the 06042026
bag. Harnesses live in the test-ws repo (`scripts/dliio_{asan,tsan,gdb}_*`);
narrative there: `results/SESSION_2026-07-02_dliio_crash_hunt.md`.

## Headline
**The crash reproduces on the current (guarded) code — the blocker is NOT closed,
and it is a data race, not the mechanism be78056 claimed.**

## Evidence

| Instrument | Reps | Crashes | Note |
|---|---|---|---|
| native (FINDINGS_2026-06-26) | 24 | 6 (~25%) | baseline |
| ASan+UBSan | 12 | 2 (~1/6) | reproduces; canNOT localize |
| gdb (break `std::__throw_out_of_range_fmt`) | 15 | **0** | crash vanishes |

Same signature each time: `std::out_of_range vector::_M_range_check`, garbage
**target-cloud-sized** index (128229969, 18446744073683142083 — tiny-float /
sign-extended bit patterns), firing immediately after `degenerate along 6
direction(s)`.

## What this means

1. **be78056 / e05ee3b guards do not cover the crashing `.at()`.** They guard
   `linearize`'s `target_index` (nano_gicp.cc ~1456) and
   `estimate_spatial_intensity_gradient` (~848); code inspection shows every other
   `.at()` in nano_gicp.cc is a bounded loop-counter. So the throw is a racing
   out-of-bounds WRITER scribbling a target-sized buffer (correspondences_ / a
   covariance or index vector), consistent with the garbage-index bit patterns.

2. **VERIFICATION_2026-06-27 §1's "one ASan rep on the bag will name the writer"
   is INSUFFICIENT.** The crash is `vector::at()`'s bounds-CHECK throwing — a C++
   exception, not a raw bad access — so ASan sees nothing at the read, and the
   write lands in live memory (no redzone). `handle_abort=1` hit an ASan
   nested-signal and produced no backtrace. (The §1 kd-tree fail-safe analysis is
   still correct; it just doesn't reach the actual writer.)

3. **It is a heisenbug / data race.** gdb suppressed it 0/15 (P≈1.3% if the rate
   were 25%); TSan never reached deg=6 at RATE 1.0 or 0.3 (its slowdown perturbs
   the timing that drives the collapse). Only native/ASan timing reaches the
   crash. See FINDINGS_2026-07-02_tsan_races.md — the 79 TSan reports are unrelated
   false positives (single-group members, uninstrumented rclcpp), NOT this writer.

## Recommended next step
**Un-guard + ASan** is the one instrument that still reaches deg=6 AND can localize:
temporarily remove the be78056 read-side guard in `linearize` so the garbage-index
access becomes a raw OOB read that ASan traps as heap-buffer-overflow (with
`handle_segv`), naming the corrupted buffer + its allocation site — which points at
the writer. Then audit the deg=6-only code paths (governor / X-ICP degeneracy
handling) for the unsynchronized write to a target-sized buffer.
