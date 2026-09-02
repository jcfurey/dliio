# Current state: `lidar_image`/`imageRefs` crash, uncommitted fix (2026-07-09)

Quick-reference for picking this back up. Full narrative:
`doc/FINDINGS_2026-07-09_07052026_rviz.md`. Background/motivation for the
`imageRefs` feature itself: `doc/INTENSITY_AUDIT_2026-07-09.md`.

## Working tree (uncommitted)

```
 M src/dlio/odom.cc
?? doc/FINDINGS_2026-07-09_07052026_rviz.md
?? doc/STATUS_2026-07-09_lidarimg_crash_hunt.md   (this file)
?? launch/dlio_07052026_notunnel.launch.py
```

`src/dlio/odom.cc` diff (in `buildSubmap()`, ~line 3620-3660): two changes,
both kept but **not proven to fix the crash**:

1. Moved `build_visual_refs`/`build_lidar_refs`'s `.size()` comparisons
   (`this->keyframe_{visual,lidar}_refs.size() == this->keyframes.size()`)
   inside a `keyframes_mutex` lock. Previously unlocked, racing
   `updateKeyframes()`'s locked `push_back` on the same vectors from the main
   thread — a real bug (unlocked read racing a push_back-triggered
   reallocation is UB) independent of whether it's THIS crash's cause. **Keep
   this regardless of what the crash hunt finds.**
2. Hardened `this->keyframe_lidar_refs[k]` → `.at(k)` so an out-of-range `k`
   throws immediately with a precise index/size instead of silently reading
   past the end and propagating a garbage `shared_ptr` downstream.

`launch/dlio_07052026_notunnel.launch.py` — new isolation launch file (plain
`dlio.yaml` + extrinsics, no tunnel overlay), used to test whether
`ouster_tunnel.yaml`'s tuned params were the source of a separate divergence
finding on this bag. Independent of the crash investigation; safe to commit
whenever the RViz-harness scripts (`resple_test_ws/scripts/dliio_07052026_rviz.sh`)
are.

## The crash

`use_lidar_image:=true` on the raw-packet `07052026_4_an` bag (live RViz,
`dlio_07052026.launch.py`) reliably crashes `dlio_odom_node` with
`std::out_of_range` / `vector::_M_range_check`, SIGABRT. Two reproductions so
far, BOTH with implausibly large `__n`/size numbers (hundreds of millions to
quintillions — no real container in this pipeline is that big), consistent
with reading a corrupted/dangling vector control block rather than a
legitimate off-by-one:

| rep | time to crash | `__n` | `size()` |
|---|---|---|---|
| pre-fix | ~108s | 18446744072558204407 | 995238912 |
| post-fix | ~10 min | 1972129020 | 381691607 |

The post-fix rep crashed LATER and with different numbers, but the same
signature — inconclusive on whether fix #1 above helped, made no difference,
or the real bug is a different (of ~15 candidate) `.at()`/`->at()` call site
in the `lidar_image` path (`accumulateLidarMapResidual`'s `target_->at(j)`,
`sampleKeyframeLidarRefs`'s `cloud->at(i)`, several
`organized->at(col,row)` in the image-build functions — full list in the
FINDINGS doc).

## Next step (not started)

Neither crash log includes a stack trace — static reading has hit
diminishing returns. Run the ASan/gdb combo-hunt harness
(`resple_test_ws/scripts/dliio_asan_combo_hunt.sh`,
`resple_test_ws/scripts/dliio_gdb_hunt.sh` — same tooling used for the
still-open 2026-06-25 photometric-loop crash) against this exact repro
(`use_lidar_image:=true`, bag `07052026_4_an`, `dlio_07052026.launch.py`) to
get a definitive stack trace before attempting another fix.

## Do not

- Don't trust `odom/lidar_image/imageRefs: true` / `use_lidar_image:=true` on
  any bag until this is root-caused.
- Don't re-run this exact repro expecting a different outcome without either
  the ASan/gdb trace or a new code change — it has now failed identically
  (modulo garbage-number specifics) twice.
