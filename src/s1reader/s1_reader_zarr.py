"""
Reader for EOPF-CPM zarr Sentinel-1 SLC products.

Returns Sentinel1BurstSlc objects compatible with compass/s1reader.
SLC data is served directly from zarr via GDAL /vsimem/ — no tiff written.

Usage
-----
    from s1reader.s1_reader_zarr import load_bursts_from_zarr

    bursts = load_bursts_from_zarr(
        zarr_path   = "/path/to/S1A_IW_SLC.zarr",
        orbit_path  = "/path/to/S1A_OPER_AUX_POEORB_...EOF",
        swath_num   = 1,
        pol         = "vv",
    )
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

try:
    import zarr
except ImportError:
    raise ImportError("zarr is required: pip install zarr")

try:
    import isce3
except ImportError:
    raise ImportError("isce3 is required")

from osgeo import gdal
from packaging import version

from . import s1_annotation
from .s1_burst_id import S1BurstId
from .s1_burst_slc import Doppler, Sentinel1BurstSlc
from .s1_reader import get_osv_list_from_orbit, get_burst_orbit

# S1 C-band centre frequency from SAFE annotation (radarFrequency element).
# The EOPF zarr does not store this field, so we use the nominal value.
# It is identical for S1A/B/C/D: verified from SAFE XML radarFrequency elements.
_S1_RADAR_FREQ = 5.40500045433435e9  # Hz
_C = isce3.core.speed_of_light        # m/s

# EOPF zarr time-unit multipliers to seconds
_UNIT_TO_S = {
    "nanoseconds": 1e-9,
    "microseconds": 1e-6,
    "milliseconds": 1e-3,
    "seconds": 1.0,
}


# ── time helpers ─────────────────────────────────────────────────────────────

def _decode_time(raw: float, units: str) -> datetime.datetime:
    """Decode a scalar offset using CF 'unit since epoch' string."""
    unit, epoch_str = units.split(" since ")
    epoch = datetime.datetime.fromisoformat(epoch_str.strip())
    return epoch + datetime.timedelta(seconds=float(raw) * _UNIT_TO_S[unit.strip()])


def _decode_time_array(arr, attrs: dict) -> list[datetime.datetime]:
    units = attrs["units"]
    return [_decode_time(float(v), units) for v in arr]


# ── zarr name parser ─────────────────────────────────────────────────────────

def _parse_burst_name(name: str) -> dict:
    """Parse EOPF-CPM burst group name.

    S01SIWSLC_YYYYMMDDTHHMMSS_RRRR_AAAA_DDDD_PPPPPP_POL_IWX_BBBBBBB
    e.g. S01SIWSLC_20240205T051225_0028_A300_454B_0656D8_VH_IW1_459340
    """
    p = name.split("_")
    return {
        "relative_orbit": int(p[2]),
        "abs_orbit_hex":  p[3],
        "polarization":   p[6].lower(),
        "subswath":       p[7],          # IW1 / IW2 / IW3
        "esa_burst_id":   int(p[8]),
    }


# ── isce3 object builders ────────────────────────────────────────────────────

def _build_orbit(orbit_path: str,
                 sensing_start: datetime.datetime,
                 sensing_stop: datetime.datetime) -> isce3.core.Orbit:
    osv_list = get_osv_list_from_orbit(orbit_path)
    return get_burst_orbit(sensing_start, sensing_stop, osv_list)


def _poly1d(coeffs: np.ndarray, t0_slant_s: float) -> isce3.core.Poly1d:
    """isce3.Poly1d from 3-element coeff array and slant-range-time origin."""
    r0 = t0_slant_s * _C / 2.0
    return isce3.core.Poly1d(list(coeffs.astype(float)), r0, _C / 2.0)


def _build_doppler(poly1d: isce3.core.Poly1d,
                   starting_range: float,
                   range_pixel_spacing: float,
                   shape: tuple,
                   azimuth_time_interval: float,
                   sensing_start: datetime.datetime) -> Doppler:
    n_lines, n_samples = shape
    slant_ranges = starting_range + np.arange(n_samples) * range_pixel_spacing
    az_times = np.array([0.0, n_lines * azimuth_time_interval])
    freq_1d = np.array([poly1d.eval(r) for r in slant_ranges], dtype=np.float64)
    data_2d = np.vstack([freq_1d, freq_1d])
    lut2d = isce3.core.LUT2d(
        slant_ranges, az_times, data_2d,
        isce3.core.dataInterpMethod.BILINEAR,
    )
    return Doppler(poly1d, lut2d)


# ── annotation objects from zarr arrays ─────────────────────────────────────

def _build_calibration(cal_group) -> Optional[s1_annotation.BurstCalibration]:
    line_arr = cal_group["line"][:]
    valid = line_arr >= 0
    if not np.any(valid):
        return None
    idx = int(np.where(valid)[0][0])
    azt_attrs = dict(cal_group["azimuth_time"].attrs)
    azimuth_time = _decode_time(float(cal_group["azimuth_time"][idx]),
                                azt_attrs["units"])
    return s1_annotation.BurstCalibration(
        basename_cads="zarr",
        azimuth_time=azimuth_time,
        line=float(line_arr[idx]),
        pixel=cal_group["pixel"][:].astype(float),
        sigma_naught=cal_group["sigma_nought"][idx].astype(float),
        beta_naught=cal_group["beta_nought"][idx].astype(float),
        gamma=cal_group["gamma"][idx].astype(float),
        dn=cal_group["dn"][idx].astype(float),
    )


def _build_noise(nr_group, na_group,
                 first_valid_line: int, last_valid_line: int,
                 shape: tuple) -> Optional[s1_annotation.BurstNoise]:
    # Range noise — find first valid slot
    nr_azt_raw = nr_group["azimuth_time"][:]
    nr_azt_attrs = dict(nr_group["azimuth_time"].attrs)
    valid_r = ~np.isnan(nr_azt_raw.astype(float))
    if not np.any(valid_r):
        return None
    idx_r = int(np.where(valid_r)[0][0])
    range_azimuth_time = _decode_time(float(nr_azt_raw[idx_r]), nr_azt_attrs["units"])
    range_pixel = nr_group["pixel"][:].astype(float)
    range_lut   = nr_group["noise_range_lut"][idx_r].astype(float)

    # Azimuth noise — clip to burst line range
    na_line = na_group["line"][:]
    na_lut  = na_group["noise_azimuth_lut"][:]
    mask = (na_line >= first_valid_line) & (na_line <= last_valid_line)
    if np.any(mask):
        az_line = (na_line[mask] - first_valid_line).astype(float)
        az_lut  = na_lut[mask].astype(float)
    else:
        az_line = np.array([0.0, float(shape[0] - 1)])
        az_lut  = np.ones(2)

    return s1_annotation.BurstNoise(
        basename_nads="zarr",
        range_azimuth_time=range_azimuth_time,
        range_line=0.0,
        range_pixel=range_pixel,
        range_lut=range_lut,
        azimuth_first_azimuth_line=int(az_line[0]),
        azimuth_last_azimuth_line=int(az_line[-1]),
        azimuth_first_range_sample=0,
        azimuth_last_range_sample=shape[1] - 1,
        azimuth_line=az_line,
        azimuth_lut=az_lut,
        line_from=first_valid_line,
        line_to=last_valid_line,
    )


def _build_extended_coeffs(
    fm_coeffs: np.ndarray, fm_t0s: np.ndarray, fm_times: list,
    dc_coeffs: np.ndarray, dc_t0s: np.ndarray, dc_times: list,
    sensing_start: datetime.datetime,
) -> s1_annotation.BurstExtendedCoeffs:
    def rel_s(times):
        return np.array([(t - sensing_start).total_seconds() for t in times])

    return s1_annotation.BurstExtendedCoeffs(
        fm_rate_aztime_vec=rel_s(fm_times),
        fm_rate_coeff_arr=fm_coeffs.copy(),
        fm_rate_tau0_vec=fm_t0s.copy(),
        dc_aztime_vec=rel_s(dc_times),
        dc_coeff_arr=dc_coeffs.copy(),
        dc_tau0_vec=dc_t0s.copy(),
    )


# ── zarr-backed burst class ───────────────────────────────────────────────────
#
# Sentinel1BurstSlc is frozen=True, so we cannot add mutable fields to a subclass.
# We use a module-level registry dict keyed by id(burst) to store the zarr array.

_ZARR_SLC_REGISTRY: dict = {}


@dataclass(frozen=True)
class ZarrBurstSlc(Sentinel1BurstSlc):
    """Sentinel1BurstSlc subclass that reads SLC data directly from zarr.

    Overrides slc_to_vrt_file / slc_to_file to use GDAL /vsimem/ in-memory
    rasters — no tiff is written to disk.

    The zarr array is stored in module-level _ZARR_SLC_REGISTRY keyed by
    id(self) because frozen dataclasses cannot hold mutable extra fields.
    Call _register(burst, zarr_arr) immediately after construction.
    """

    def _get_slc_arr(self):
        arr = _ZARR_SLC_REGISTRY.get(id(self))
        if arr is None:
            raise RuntimeError(
                "ZarrBurstSlc: zarr array not registered. "
                "Call s1_reader_zarr._register(burst, zarr_arr) after construction."
            )
        return arr

    def slc_to_vrt_file(self, out_path: str):
        """Write a VRT at out_path backed by a /vsimem/ GTiff from zarr.
        No file is created on disk.
        """
        slc = self._get_slc_arr()[:]      # (n_lines, n_samples) complex64
        n_lines, n_samples = slc.shape

        vsimem_path = f"/vsimem/zarr_slc_{id(self)}.tif"
        ds = gdal.GetDriverByName("GTiff").Create(
            vsimem_path, n_samples, n_lines, 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(slc)
        ds.FlushCache()
        ds = None

        fvs, lvs = self.first_valid_sample, self.last_valid_sample
        fvl, lvl = self.first_valid_line,   self.last_valid_line
        outl, outw = self.shape

        vrt = (
            f'<VRTDataset rasterXSize="{outw}" rasterYSize="{outl}">\n'
            f'  <VRTRasterBand dataType="CFloat32" band="1">\n'
            f'    <NoDataValue>0.0</NoDataValue>\n'
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="0">{vsimem_path}</SourceFilename>\n'
            f'      <SourceBand>1</SourceBand>\n'
            f'      <SourceProperties RasterXSize="{n_samples}" RasterYSize="{n_lines}"'
            f' DataType="CFloat32"/>\n'
            f'      <SrcRect xOff="{fvs}" yOff="{fvl}"'
            f' xSize="{lvs-fvs+1}" ySize="{lvl-fvl+1}"/>\n'
            f'      <DstRect xOff="{fvs}" yOff="{fvl}"'
            f' xSize="{lvs-fvs+1}" ySize="{lvl-fvl+1}"/>\n'
            f'    </SimpleSource>\n'
            f'  </VRTRasterBand>\n'
            f'</VRTDataset>'
        )
        with open(out_path, "w") as f:
            f.write(vrt)

    def slc_to_file(self, out_path: str, fmt: str = "GTiff"):
        """Write zarr SLC directly to a file (used by rdr2geo workflow)."""
        slc = self._get_slc_arr()[:]
        n_lines, n_samples = slc.shape
        ds = gdal.GetDriverByName(fmt).Create(
            out_path, n_samples, n_lines, 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(slc)
        ds.FlushCache()
        ds = None


def _register(burst: ZarrBurstSlc, zarr_arr) -> None:
    """Associate a zarr SLC array with a ZarrBurstSlc instance."""
    _ZARR_SLC_REGISTRY[id(burst)] = zarr_arr


# ── iw2_mid_range helper ─────────────────────────────────────────────────────

def _iw2_mid_range(z) -> float:
    """Compute iw2_mid_range from any IW2 burst in the zarr store."""
    iw2_keys = sorted(k for k in z.keys() if "_IW2_" in k)
    if not iw2_keys:
        warnings.warn("No IW2 bursts found — iw2_mid_range set to 0 (bistatic correction off)")
        return 0.0
    g = z[iw2_keys[0]]
    di = dict(g.attrs).get("other_metadata", {}).get("downlink_information", {})
    fs   = float(di.get("sampling_frequency_after_decimation", 64.345e6))
    swst = float(di.get("swst_value", 0.005646))
    rps  = _C / (2.0 * fs)
    n_s  = g["measurements/slc"].shape[1]
    return swst * _C / 2.0 + 0.5 * n_s * rps


# ── per-burst polynomial extraction ─────────────────────────────────────────

def _nearest_valid_poly(poly_group, coeff_key: str, t0_key: str,
                        az_mid: datetime.datetime):
    """Return (poly1d, all_valid_coeffs, all_valid_t0s, all_valid_times)."""
    azt_attrs = dict(poly_group["azimuth_time"].attrs)
    azt_raw   = poly_group["azimuth_time"][:]
    coeffs    = poly_group[coeff_key][:]
    t0s       = poly_group[t0_key][:]

    valid = ~np.isnan(coeffs[:, 0])
    times  = [_decode_time(float(v), azt_attrs["units"])
              for v, ok in zip(azt_raw, valid) if ok]
    cv = coeffs[valid]
    tv = t0s[valid]

    nearest = int(np.argmin([abs((t - az_mid).total_seconds()) for t in times]))
    return _poly1d(cv[nearest], float(tv[nearest])), cv, tv, times


# ── public API ───────────────────────────────────────────────────────────────

def load_bursts_from_zarr(
    zarr_path: str,
    orbit_path: str,
    swath_num: int,
    pol: str = "vv",
    burst_ids: Optional[list] = None,
) -> list[ZarrBurstSlc]:
    """Load bursts from an EOPF-CPM zarr Sentinel-1 SLC product.

    Parameters
    ----------
    zarr_path : str
        Path to the .zarr directory.
    orbit_path : str
        Path to a POEORB or RESORB .EOF file.
        The zarr stores only 1 state vector; a full orbit file is required.
    swath_num : int
        Subswath {1, 2, 3}.
    pol : str
        Polarization {'vv', 'vh', 'hh', 'hv'}.
    burst_ids : list[str | S1BurstId], optional
        If given, only return matching bursts.

    Returns
    -------
    list[ZarrBurstSlc]
    """
    if swath_num not in (1, 2, 3):
        raise ValueError("swath_num must be 1, 2, or 3")
    pol = pol.lower()
    swath_tag = f"IW{swath_num}"

    z = zarr.open(str(zarr_path), mode="r")

    keys = sorted(k for k in z.keys() if f"_{pol.upper()}_{swath_tag}_" in k)
    if not keys:
        warnings.warn(f"No bursts for pol={pol} swath=IW{swath_num} in {zarr_path}")
        return []

    # Mission / frequency / wavelength from filename
    mission = Path(zarr_path).stem[:3].upper()
    radar_freq = _S1_RADAR_FREQ
    wavelength = _C / radar_freq

    # iw2_mid_range from any IW2 burst (needed for bistatic delay)
    iw2_mid = _iw2_mid_range(z)

    bursts = []
    for key in keys:
        g = z[key]
        omd = dict(g.attrs).get("other_metadata", {})
        di  = omd.get("downlink_information", {})
        pi  = omd.get("processing_information", {})
        rp  = pi.get("range_processing", {})
        ap  = pi.get("azimuth_processing", {})
        parsed = _parse_burst_name(key)

        # IPF version
        ipf_str = "003.71"
        for h in omd.get("history", []):
            v = h.get("version")
            if v:
                ipf_str = v
                break
        ipf_version = version.parse(ipf_str)

        # SLC
        slc_arr = g["measurements/slc"]
        shape   = slc_arr.shape   # (n_lines, n_samples)

        # Azimuth timing
        azt_arr   = g["measurements/azimuth_time"]
        azt_attrs = dict(azt_arr.attrs)
        azt_raw   = azt_arr[:]
        sensing_start = _decode_time(float(azt_raw[0]),  azt_attrs["units"])
        sensing_stop  = _decode_time(float(azt_raw[-1]), azt_attrs["units"])
        azimuth_time_interval = float(azt_raw[1] - azt_raw[0]) * _UNIT_TO_S[azt_attrs["units"].split(" since ")[0].strip()]

        # Range parameters
        fs               = float(di.get("sampling_frequency_after_decimation", 64.345e6))
        range_pixel_spacing = _C / (2.0 * fs)
        slant_range_time = float(di.get("swst_value", 1.043e-4))
        starting_range   = slant_range_time * _C / 2.0
        range_bandwidth  = float(rp.get("look_bandwidth", rp.get("total_bandwidth", 56.5e6)))
        range_window_coefficient = float(rp.get("window_coefficient", 0.75))
        range_window_type = "HAMMING"

        rank            = int(di.get("rank", 9))
        prf_raw_data    = float(di.get("prf", 1717.0))
        range_chirp_rate = float(di.get("tx_pulse_ramp_rate", 1.078e12))
        azimuth_steer_rate = float(omd.get("azimuth_steering_rate", 0.0))
        # fallback average azimuth pixel spacing for S1 IW ~14 m
        average_azimuth_pixel_spacing = 13.9

        # Orbit from EOF
        orbit = _build_orbit(orbit_path, sensing_start, sensing_stop)
        orbit_direction = "ASCENDING"  # override via orbit if needed

        # Mid-burst azimuth time for polynomial selection
        az_mid = sensing_start + datetime.timedelta(
            seconds=shape[0] * azimuth_time_interval / 2.0)

        # FM rate
        fm_poly1d, fm_cv, fm_tv, fm_times = _nearest_valid_poly(
            g["conditions/azimuth_fm_rate"],
            "azimuth_fm_rate_polynomial", "t0", az_mid)

        # Doppler centroid
        dc_poly1d, dc_cv, dc_tv, dc_times = _nearest_valid_poly(
            g["conditions/doppler_centroid"],
            "data_dc_polynomial", "t0", az_mid)

        doppler = _build_doppler(dc_poly1d, starting_range, range_pixel_spacing,
                                  shape, azimuth_time_interval, sensing_start)

        # Burst ID
        burst_id = S1BurstId(
            track_number=parsed["relative_orbit"],
            esa_burst_id=parsed["esa_burst_id"],
            subswath=swath_tag.lower(),
        )

        # Valid pixel region — zarr doesn't store firstValidSample arrays,
        # use full burst extent
        first_valid_line   = 0
        last_valid_line    = shape[0] - 1
        first_valid_sample = 0
        last_valid_sample  = shape[1] - 1

        # Border & center from GCPs
        gcp  = g["conditions/gcp"]
        lats = gcp["latitude"][:]
        lons = gcp["longitude"][:]
        vg   = ~np.isnan(lats)
        if np.any(vg):
            center = (float(np.nanmean(lons)), float(np.nanmean(lats)))
            border = list(zip(lons[vg].tolist(), lats[vg].tolist()))
        else:
            center, border = (0.0, 0.0), []

        # Calibration & noise
        burst_calibration = _build_calibration(g["quality/calibration"])
        burst_noise = _build_noise(
            g["quality/noise_range"], g["quality/noise_azimuth"],
            first_valid_line, last_valid_line, shape,
        )

        # EAP — not available from zarr (needs AUX_CAL)
        # Build a minimal BurstEAP with empty arrays so compass can skip it
        dummy_arr = np.zeros(1)
        burst_eap = s1_annotation.BurstEAP(
            freq_sampling=fs,
            eta_start=sensing_start,
            tau_0=slant_range_time,
            tau_sub=dummy_arr,
            theta_sub=dummy_arr,
            azimuth_time=sensing_start,
            ascending_node_time=sensing_start,
            gain_eap=dummy_arr,
            delta_theta=0.0,
        )

        # Extended coefficients for FM rate mismatch mitigation
        extended_coeffs = _build_extended_coeffs(
            fm_cv, fm_tv, fm_times,
            dc_cv, dc_tv, dc_times,
            sensing_start,
        )

        burst_rfi_info = SimpleNamespace(
            rfi_mitigation_performed="false",
            rfi_mitigation_domain="",
            rfi_burst_report={},
        )

        burst_misc_metadata = SimpleNamespace(
            processing_info_dict={},
            azimuth_looks=int(ap.get("number_of_looks", 1)),
            slant_range_looks=int(rp.get("number_of_looks", 1)),
            inc_angle_near_range=float(
                np.nanmean(g["conditions/antenna_pattern/incidence_angle"][:, 0])),
            inc_angle_far_range=float(
                np.nanmean(g["conditions/antenna_pattern/incidence_angle"][:, -1])),
        )

        burst = ZarrBurstSlc(
            ipf_version=ipf_version,
            sensing_start=sensing_start,
            radar_center_frequency=radar_freq,
            wavelength=wavelength,
            azimuth_steer_rate=azimuth_steer_rate,
            average_azimuth_pixel_spacing=average_azimuth_pixel_spacing,
            azimuth_time_interval=azimuth_time_interval,
            slant_range_time=slant_range_time,
            starting_range=starting_range,
            iw2_mid_range=iw2_mid,
            range_sampling_rate=fs,
            range_pixel_spacing=range_pixel_spacing,
            shape=shape,
            azimuth_fm_rate=fm_poly1d,
            doppler=doppler,
            range_bandwidth=range_bandwidth,
            polarization=pol.upper(),
            burst_id=burst_id,
            platform_id=mission,
            safe_filename=Path(zarr_path).stem,
            center=center,
            border=border,
            orbit=orbit,
            orbit_direction=orbit_direction,
            abs_orbit_number=int(parsed["abs_orbit_hex"], 16),
            tiff_path="",                 # empty — zarr-backed
            i_burst=0,
            first_valid_sample=first_valid_sample,
            last_valid_sample=last_valid_sample,
            first_valid_line=first_valid_line,
            last_valid_line=last_valid_line,
            range_window_type=range_window_type,
            range_window_coefficient=range_window_coefficient,
            rank=rank,
            prf_raw_data=prf_raw_data,
            range_chirp_rate=range_chirp_rate,
            burst_calibration=burst_calibration,
            burst_noise=burst_noise,
            burst_eap=burst_eap,
            extended_coeffs=extended_coeffs,
            burst_rfi_info=burst_rfi_info,
            burst_misc_metadata=burst_misc_metadata,
        )
        _register(burst, slc_arr)
        bursts.append(burst)

    # Optional filter
    if burst_ids:
        ids_norm = {str(b) if isinstance(b, S1BurstId) else b for b in burst_ids}
        bursts = [b for b in bursts if str(b.burst_id) in ids_norm]

    return bursts
