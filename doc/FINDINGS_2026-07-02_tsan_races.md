# TSan findings — combo run, 2026-07-02

TSan (`DLIIO_SANITIZE=thread`) run of the `combo` config (X-ICP ternary fullRatio
0.05 + governor 0.15, LiDAR-only) on the 06042026 bag, chasing the
FINDINGS_2026-06-26 `std::out_of_range` deployment blocker. Harness in the
test-ws repo: `scripts/dliio_tsan_combo_hunt.sh` (+ `dliio_asan_combo_hunt.sh`,
`dliio_gdb_hunt.sh`). Full crash-hunt narrative:
`resple_test_ws/results/SESSION_2026-07-02_dliio_crash_hunt.md`.

## Two separate things, do not conflate

### 1. The deg=6 crash is NOT here (still open)
The blocker crash corrupts a **target-cloud-sized** gicp buffer (millions of
bytes, its own heap allocation) and only fires at **deg=6**. TSan never reached
deg=6 — its ~10x slowdown perturbs timing so the node tracks (or drops scans)
without collapsing (deg6=0 at RATE 1.0 AND 0.3), exactly as gdb suppressed the
crash 0/15. So TSan could not observe the crash race. That bug remains for the
un-guard+ASan approach (ASan is the only instrument that still reaches deg=6).

### 2. The 79 races TSan DID report are almost certainly FALSE POSITIVES
49 race blocks, all on **one** heap block — the `OdomNode` object itself (~11.6 KB,
allocated in the ctor, odom.cc:358). Three clusters:

| Cluster | Members | Sole accessor | Group |
|---|---|---|---|
| A | `odom_ros` / `pose_ros` (message buffers) | ctor + `publishPose` (odom.cc:942–978) | publish **timer** |
| B | `transform_prev_stamp_`, `ang_vel_cg_prev_`, `transform_imu_init_` | `transformImu` (odom.cc:3027–3062, called only from `callbackImu`) | **IMU** cb |
| C | `calib_gyro_avg_`, `calib_accel_avg_`, `calib_num_samples_` | `callbackImu` calibration branch (odom.cc:2056–2084) | **IMU** cb |

**Every raced member is touched by exactly ONE MutuallyExclusive callback group**
(verified by grep — no second accessor). A MutuallyExclusive group cannot run its
callbacks concurrently, so these members are **never genuinely accessed in
parallel**. TSan flags them only because **rclcpp/rcl are not compiled with TSan**
(only dliio is): the executor's callback-group synchronization — the
acquire/release that establishes happens-before between successive callbacks that
happen to land on *different* executor worker threads — is invisible to TSan.
TSan therefore sees "member written by thread T31, later by T44, no sync between
them" and reports a race, even though the two callbacks never overlapped.

Corroborating evidence that this is instrumentation blindness, not a real bug:
- **Genuinely cross-group shared state is already correctly locked and TSan did
  NOT flag it:** `state` / `imu_stamp` (IMU writer vs `publishPose` timer reader)
  under `geo.mtx`; `imu_rates` (IMU writer vs scan-thread `publishDiagnostics`
  reader) under `mtx_imu`. Those `std::mutex`es live in dliio code → TSan tracks
  them → no report. Only the *unlocked, single-group* members show up.
- The `libgomp`-not-annotated caveat (RESPLE `HARDENING.md`) is the same class of
  problem for OpenMP; here it is the uninstrumented rclcpp executor.

Residual caveat: "false positive" holds on x86/TSO and if rclcpp's group sync is a
proper acquire/release (it uses a seq_cst atomic exchange on `can_be_taken_from_`,
which is). On a weakly-ordered arch (aarch64) a *missing* barrier would be a real
(if rare) bug — so the defensive fix has portability value even if it is a no-op
on x86.

## Action taken
- **Did NOT add locks** to silence these — that would add hot-path cost (IMU cb,
  100 Hz publish timer) for a non-race on the x86 deployment target.
- **Added `test/tsan_suppressions.txt`** (this repo) scoped to the two
  single-purpose functions `transformImu` and `publishPose`. `callbackImu` is left
  UNsuppressed on purpose so its (benign, first-~3 s) calibration-accumulator races
  stay visible and a future genuine race there is not masked. Validated: on a rerun
  the reports dropped **49 → 3**, the 3 being exactly the intentionally-visible
  callbackImu calibration races — `transformImu`/`publishPose` gone, and **no
  genuine cross-group race surfaced from underneath.** Wire it in with
  `TSAN_OPTIONS=suppressions=<path>` (the test-ws hunt harness already does).
- Still open, optional & defensive-for-aarch64 only: local-state refactor of
  `transformImu` (locals or `std::atomic`) and publish-from-locals in
  `publishPose` — low value on x86, removes the reports by construction.

## Genuine bug found regardless
`publishPose` snapshots `state`/`imu_stamp` under `geo.mtx` correctly (932–937),
then writes the shared member `odom_ros`/`pose_ros` outside the lock — fine *because*
it is timer-only. No action needed unless another group is ever made to publish.
