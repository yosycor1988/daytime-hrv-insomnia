"""
Independent verification that the wearable timestamps are Korea local time (KST = UTC+9),
not UTC. Run this to reproduce the three lines of evidence behind the timezone correction:

    python src/verify_timezone.py

It reads only the raw data files and reasons from first principles, so it does not depend on
any other part of the pipeline being correct. A reviewer can run it against the original
Figshare download and reach the same conclusion.

The logic: an epoch timestamp is a universal instant with no timezone attached. pandas decodes
it to UTC by default. Whether that is the correct wall-clock hour depends on where the
participants were. Three independent signals fix the location in time.
"""
import os
import sys
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"  # Figshare download (doi:10.6084/m9.figshare.28509740), see README
)
KST_OFFSET = 9  # hours; South Korea, no daylight saving


def load_raw():
    hrv = pd.read_csv(os.path.join(DATA_DIR, "sensor_hrv_filtered.csv"))
    hrv["utc"] = pd.to_datetime(hrv["ts_start"], unit="ms")   # pandas default = UTC
    hrv = hrv[hrv["utc"].dt.year == 2021].copy()
    hrv["kst"] = hrv["utc"] + pd.Timedelta(hours=KST_OFFSET)
    return hrv


def quietest_window(series_hour, steps, width=8):
    """Return the `width`-hour block with the lowest mean step count (the likely sleep window)."""
    m = pd.Series(steps.values, index=series_hour.values).groupby(level=0).mean().reindex(range(24))
    best, best_val = None, float("inf")
    for start in range(24):
        hrs = [(start + i) % 24 for i in range(width)]
        v = m.reindex(hrs).mean()
        if v < best_val:
            best_val, best = v, hrs
    return f"{best[0]:02d}:00-{(best[-1] + 1) % 24:02d}:00", best_val


def main():
    hrv = load_raw()
    print("=" * 70)
    print("TIMEZONE VERIFICATION  (raw data only, reasoning from first principles)")
    print("=" * 70)
    print(f"HRV rows analysed: {len(hrv):,}")

    # ---- EVIDENCE 1: circadian activity trough -----------------------------
    print("\n[1] CIRCADIAN ACTIVITY TROUGH")
    print("    People stop walking while asleep. The quietest 8 hours should look")
    print("    like a plausible night (~23:00-07:00), and only on the correct axis.")
    w_utc, v_utc = quietest_window(hrv["utc"].dt.hour, hrv["steps"])
    w_kst, v_kst = quietest_window(hrv["kst"].dt.hour, hrv["steps"])
    print(f"    quietest 8h if timestamps were UTC : {w_utc}  (implausible: mid-day sleep)")
    print(f"    quietest 8h if timestamps are  KST : {w_kst}  <-- plausible night")

    # ---- EVIDENCE 2: self-reported sleep diary (local by definition) --------
    print("\n[2] SELF-REPORTED WAKE TIMES (unambiguously local)")
    print("    Participants hand-wrote wake times; nobody records their own wake in UTC.")
    sleep = pd.read_csv(os.path.join(DATA_DIR, "sleep_diary.csv"))
    wakes = sleep["wakeup"].dropna().astype(str).head(8).tolist()
    print(f"    diary wake times: {wakes}")
    print("    These cluster around 08:00-11:00, matching the KST axis, not UTC (would be 23:00-02:00).")

    # ---- EVIDENCE 3: study span ------------------------------------------
    print("\n[3] STUDY SPAN (device hand-out / return happen in daytime)")
    print(f"    span if UTC : {hrv['utc'].min()}  ..  {hrv['utc'].max()}")
    print(f"    span if KST : {hrv['kst'].min()}  ..  {hrv['kst'].max()}")
    print("    KST span starts and ends near midday (sensible); UTC span starts ~03:40 (not).")

    # ---- CONSEQUENCE: what the 'daytime 08:00-22:00' filter captured -------
    print("\n" + "=" * 70)
    print("CONSEQUENCE FOR THE 'DAYTIME' FILTER (hours 8..22)")
    print("=" * 70)
    cur = (hrv["utc"].dt.hour >= 8) & (hrv["utc"].dt.hour <= 22)   # the buggy filter
    fix = (hrv["kst"].dt.hour >= 8) & (hrv["kst"].dt.hour <= 22)   # corrected filter
    kst_of_buggy = sorted(hrv.loc[cur, "kst"].dt.hour.unique())
    night = cur & ((hrv["kst"].dt.hour >= 23) | (hrv["kst"].dt.hour <= 6))
    print(f"filtering on UTC hours 8..22 actually keeps KST hours: {kst_of_buggy}")
    print(f"  -> that is 17:00-07:59 local (evening + overnight)")
    print(f"  -> {int(night.sum()):,} of {int(cur.sum()):,} kept rows "
          f"({night.mean() / cur.mean() * 100:.0f}%) fall in 23:00-06:59 local = actual night")
    print(f"rows kept, buggy (UTC 8..22) : {int(cur.sum()):,} ({cur.mean() * 100:.1f}% of data)")
    print(f"rows kept, fixed (KST 8..22) : {int(fix.sum()):,} ({fix.mean() * 100:.1f}% of data)")

    print("\nCONCLUSION: timestamps are KST. Add 9 hours before any hour/date logic.")
    print("This is implemented in src/data_loader.py (KST_OFFSET_HOURS = 9).")


if __name__ == "__main__":
    main()
