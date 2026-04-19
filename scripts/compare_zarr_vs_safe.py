#!/usr/bin/env python
"""
Standalone test: compare EOPF-CPM zarr vs SAFE burst fields used by compass/isce3.

Downloads a Sentinel-1 SAFE zip, EOPF zarr, and orbit file automatically,
loads both burst objects, and prints a field-by-field comparison of every
attribute read by compass/isce3 for geocoding.

Usage
-----
    python scripts/compare_zarr_vs_safe.py [--work-dir /tmp/zarr_test]
    python scripts/compare_zarr_vs_safe.py --skip-download --work-dir /path/to/existing

Requirements
------------
    pip install asf_search
    s1-reader with s1_reader_zarr (EOPF-CPM zarr support)

Test scene
----------
    S1A IW SLC, 2024-02-05, track 22, burst t022_045934_iw1
    Central Italy (Apennines), descending pass
"""

import argparse
import datetime
import numpy as np
from pathlib import Path

# ── fixed test scene ──────────────────────────────────────────────────────────
SCENE_ID   = "S1A_IW_SLC__1SDV_20240205T051225_20240205T051253_052419_0656D8_454B"
BURST_ID   = "t022_045934_iw1"
SWATH_NUM  = 1
POL        = "vv"

ORBIT_URL  = (
    "https://aux.sentinel1.eo.esa.int/POEORB/2024/02/25/"
    "S1A_OPER_AUX_POEORB_OPOD_20240225T070745_V20240204T225942_20240206T005942.EOF"
)
ORBIT_NAME = Path(ORBIT_URL).name

# EOPF sample service
_EOPF_BASE = (
    "https://s3.waw3-1.cloudferro.com/swift/v1/"
    "AUTH_c2073fb14d6f4d38a63e9c2e57578285/eopf-sample-service-public/s1"
)
# ─────────────────────────────────────────────────────────────────────────────


def download_safe(work_dir: Path) -> Path:
    out_path = work_dir / f"{SCENE_ID}.zip"
    if out_path.exists():
        print(f"  SAFE already present: {out_path}")
        return out_path
    try:
        import asf_search as asf
    except ImportError:
        raise ImportError("pip install asf_search")
    print(f"  Searching ASF for {SCENE_ID} ...")
    results = asf.search(granule_list=[SCENE_ID])
    if not results:
        raise ValueError(f"Scene {SCENE_ID} not found on ASF")
    print(f"  Downloading SAFE to {work_dir} ...")
    asf.download_urls(urls=[results[0].properties["url"]], path=str(work_dir))
    return out_path


def download_zarr(work_dir: Path) -> Path:
    zarr_dir = work_dir / f"{SCENE_ID}.zarr"
    if zarr_dir.exists():
        print(f"  zarr already present: {zarr_dir}")
        return zarr_dir
    zarr_url = f"{_EOPF_BASE}/{SCENE_ID}.zarr"
    import subprocess
    print(f"  Downloading zarr from {zarr_url} ...")
    subprocess.run(["python", "-m", "zarr", "copy", zarr_url, str(zarr_dir)], check=True)
    return zarr_dir


def download_orbit(work_dir: Path) -> Path:
    import urllib.request
    out_path = work_dir / ORBIT_NAME
    if out_path.exists():
        print(f"  Orbit already present: {out_path}")
        return out_path
    print(f"  Downloading orbit from ESA ...")
    urllib.request.urlretrieve(ORBIT_URL, out_path)
    return out_path


def load_safe_burst(safe_path, orbit_path):
    import s1reader
    bursts = s1reader.load_bursts(str(safe_path), str(orbit_path), SWATH_NUM, POL,
                                   burst_ids=[BURST_ID])
    if not bursts:
        raise ValueError(f"Burst {BURST_ID} not found in {safe_path}")
    return bursts[0]


def load_zarr_burst(zarr_path, orbit_path):
    from s1reader.s1_reader_zarr import load_bursts_from_zarr
    bursts = load_bursts_from_zarr(str(zarr_path), str(orbit_path), SWATH_NUM, POL,
                                    burst_ids=[BURST_ID])
    if not bursts:
        raise ValueError(f"Burst {BURST_ID} not found in {zarr_path}")
    return bursts[0]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.8g}"
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    return str(v)


def cmp_scalar(name, vs, vz, unit="", tol=None):
    sv, sz = fmt(vs), fmt(vz)
    try:
        diff = abs(float(vs) - float(vz))
        diff_str = f"{diff:.3e} {unit}".strip()
        flag = ("OK" if tol is None or diff <= tol else "WARN")
    except (TypeError, ValueError):
        if isinstance(vs, datetime.datetime) and isinstance(vz, datetime.datetime):
            diff = abs((vs - vz).total_seconds())
            diff_str = f"{diff:.6f} s"
            flag = "OK" if diff < 1e-3 else "WARN"
        else:
            diff_str = "n/a"
            flag = "OK" if sv == sz else "DIFF"
    mark = "✓" if (flag in ("OK", "") and sv == sz) else ("≈" if flag == "OK" else "≠")
    print(f"  {mark} {name:<35}  safe={sv:<30}  zarr={sz:<30}  diff={diff_str}  {flag}")


def cmp_lut2d(name, ls, lz):
    print(f"\n  [{name}]")
    for attr in ["x_start", "x_spacing", "y_start", "y_spacing"]:
        cmp_scalar(f"  {attr}", getattr(ls, attr), getattr(lz, attr))
    cmp_scalar("  data shape", ls.data.shape, lz.data.shape)
    if ls.data.shape == lz.data.shape:
        d = np.abs(ls.data - lz.data)
        print(f"    data max_abs_diff={d.max():.4e}  mean={d.mean():.4e}")


def cmp_orbit(os_, oz):
    print(f"\n  [orbit]")
    cmp_scalar("  reference_epoch", os_.reference_epoch.isoformat_usec(),
               oz.reference_epoch.isoformat_usec())
    cmp_scalar("  size", os_.size, oz.size)
    ts, tz = np.array(os_.time), np.array(oz.time)
    if ts.shape == tz.shape:
        print(f"    time max_abs_diff  = {np.abs(ts-tz).max():.4e} s")
    ps, pz = np.array(os_.position), np.array(oz.position)
    if ps.shape == pz.shape:
        print(f"    position max_diff  = {np.abs(ps-pz).max():.4e} m")


def run_checks(bs, bz):
    """Run pass/fail assertions on the most critical geocoding fields."""
    errs = []
    if str(bs.burst_id) != str(bz.burst_id):
        errs.append(f"burst_id mismatch: safe={bs.burst_id} zarr={bz.burst_id}")
    dt = abs((bs.sensing_start - bz.sensing_start).total_seconds())
    if dt >= 1e-3:
        errs.append(f"sensing_start differs by {dt:.6f} s (threshold 1 ms)")
    if bs.shape != bz.shape:
        errs.append(f"shape mismatch: safe={bs.shape} zarr={bz.shape}")
    if abs(bs.starting_range - bz.starting_range) > 1.0:
        errs.append(f"starting_range differs by {abs(bs.starting_range-bz.starting_range):.2f} m")
    if bz.doppler.lut2d.y_start == 0.0:
        errs.append("Doppler LUT2d y_start is 0.0 (azimuth times not orbit-relative)")
    if bz.first_valid_sample == 0 and bz.last_valid_sample == bz.shape[1] - 1:
        errs.append("valid sample bounds span full width (zero padding not detected)")
    return errs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default="/tmp/zarr_reader_test",
                        help="Directory for downloads and output")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download; use --safe-path/--zarr-path/--orbit-path")
    parser.add_argument("--safe-path",  help="Path to existing SAFE zip")
    parser.add_argument("--zarr-path",  help="Path to existing EOPF zarr directory")
    parser.add_argument("--orbit-path", help="Path to existing POEORB .EOF file")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Work dir: {work_dir}")
    print(f"Burst:    {BURST_ID}  IW{SWATH_NUM}  {POL.upper()}")

    if args.skip_download:
        safe_path  = Path(args.safe_path)  if args.safe_path  else work_dir / f"{SCENE_ID}.zip"
        zarr_path  = Path(args.zarr_path)  if args.zarr_path  else work_dir / f"{SCENE_ID}.zarr"
        orbit_path = Path(args.orbit_path) if args.orbit_path else work_dir / ORBIT_NAME
        for p in [safe_path, zarr_path, orbit_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
    else:
        print("\n--- Downloading inputs ---")
        safe_path  = download_safe(work_dir)
        zarr_path  = download_zarr(work_dir)
        orbit_path = download_orbit(work_dir)

    print("\n--- Loading SAFE burst ---")
    bs = load_safe_burst(safe_path, orbit_path)
    print(f"  burst_id={bs.burst_id}  shape={bs.shape}  start={bs.sensing_start}")

    print("\n--- Loading zarr burst ---")
    bz = load_zarr_burst(zarr_path, orbit_path)
    print(f"  burst_id={bz.burst_id}  shape={bz.shape}  start={bz.sensing_start}")

    print(f"\n{'='*100}")
    print(f"  Field comparison — {BURST_ID}  IW{SWATH_NUM}  {POL.upper()}")
    print(f"{'='*100}")

    print("\n[Burst scalars]")
    cmp_scalar("burst_id",              str(bs.burst_id),        str(bz.burst_id))
    cmp_scalar("sensing_start",         bs.sensing_start,         bz.sensing_start)
    cmp_scalar("sensing_stop",          bs.sensing_stop,          bz.sensing_stop)
    cmp_scalar("starting_range",        bs.starting_range,        bz.starting_range,       "m",  1.0)
    cmp_scalar("range_pixel_spacing",   bs.range_pixel_spacing,   bz.range_pixel_spacing,  "m",  1e-4)
    cmp_scalar("azimuth_time_interval", bs.azimuth_time_interval, bz.azimuth_time_interval, "s",  1e-7)
    cmp_scalar("wavelength",            bs.wavelength,            bz.wavelength,           "m",  1e-6)
    cmp_scalar("abs_orbit_number",      bs.abs_orbit_number,      bz.abs_orbit_number)
    cmp_scalar("shape",                 bs.shape,                 bz.shape)
    cmp_scalar("first_valid_sample",    bs.first_valid_sample,    bz.first_valid_sample)
    cmp_scalar("last_valid_sample",     bs.last_valid_sample,     bz.last_valid_sample)
    cmp_scalar("first_valid_line",      bs.first_valid_line,      bz.first_valid_line)
    cmp_scalar("last_valid_line",       bs.last_valid_line,       bz.last_valid_line)
    cmp_scalar("iw2_mid_range",         bs.iw2_mid_range,         bz.iw2_mid_range,        "m",  100.0)

    print(f"\n[Doppler poly1d coeffs]")
    print(f"  safe: {bs.doppler.poly1d.coeffs}")
    print(f"  zarr: {bz.doppler.poly1d.coeffs}")
    cmp_lut2d("doppler.lut2d", bs.doppler.lut2d, bz.doppler.lut2d)

    print(f"\n[FM rate (azimuth_fm_rate) Poly1d coeffs]")
    print(f"  safe: {bs.azimuth_fm_rate.coeffs}")
    print(f"  zarr: {bz.azimuth_fm_rate.coeffs}")
    cmp_scalar("  order", bs.azimuth_fm_rate.order, bz.azimuth_fm_rate.order)

    print(f"\n[Radar grid]")
    rgs, rgz = bs.as_isce3_radargrid(), bz.as_isce3_radargrid()
    cmp_scalar("  sensing_start",      rgs.sensing_start,      rgz.sensing_start,      "s")
    cmp_scalar("  prf",                rgs.prf,                rgz.prf,                "Hz")
    cmp_scalar("  starting_range",     rgs.starting_range,     rgz.starting_range,     "m")
    cmp_scalar("  range_pixel_spacing", rgs.range_pixel_spacing, rgz.range_pixel_spacing, "m")
    cmp_scalar("  length (lines)",     rgs.length,             rgz.length)
    cmp_scalar("  width (samples)",    rgs.width,              rgz.width)

    cmp_orbit(bs.orbit, bz.orbit)

    print(f"\n{'='*100}")
    print("\n--- Checks ---")
    errs = run_checks(bs, bz)
    if errs:
        for e in errs:
            print(f"  FAIL: {e}")
        raise SystemExit(1)
    print("  All critical checks PASSED")


if __name__ == "__main__":
    main()
