"""
Reader for Sentinel-1 Stripmap (SM) SLC SAFE products.

Sentinel-1 Stripmap differs from IW TOPS in several key ways:
  - No burst delineation: the entire acquisition is one contiguous swath.
    swathTiming/linesPerBurst = 0, samplesPerBurst = 0, burstList is empty.
  - Shape comes from imageAnnotation/imageInformation/numberOfLines and numberOfSamples.
  - sensing_start = productFirstLineUtcTime (from imageInformation, not burstList).
  - No azimuthSteeringRate (zero for stripmap, no TOPS beam steering).
  - iw2_mid_range: not applicable; set to mid-range of the stripmap swath itself.
  - Valid pixel bounds (first/last valid sample/line): not provided in annotation;
    derived from the geolocation grid extent or assumed to be the full swath.
  - Returns a single Sentinel1BurstSlc per polarization (i_burst=0).

Public API
----------
    load_stripmap_bursts(path, orbit_path, pol='vv')
        Load Sentinel-1 Stripmap SLC from a SAFE zip or directory.
        Returns list[Sentinel1BurstSlc] (one element per call).
"""

from __future__ import annotations

import datetime
import os
import warnings
import zipfile

import isce3
import lxml.etree as ET
import numpy as np
import shapely

from nisar.workflows.stage_dem import check_dateline
from packaging import version
from types import SimpleNamespace

from s1reader import s1_annotation
from s1reader.s1_annotation import (
    RFI_INFO_AVAILABLE_FROM,
    CalibrationAnnotation,
    AuxCal,
    BurstCalibration,
    BurstEAP,
    BurstNoise,
    BurstExtendedCoeffs,
    NoiseAnnotation,
    ProductAnnotation,
    SwathRfiInfo,
    SwathMiscMetadata,
)
from s1reader.s1_burst_slc import Doppler, Sentinel1BurstSlc
from s1reader.s1_burst_id import S1BurstId
from s1reader.s1_reader import (
    as_datetime,
    calculate_centroid,
    doppler_poly1d_to_lut2d,
    get_burst_orbit,
    get_ipf_version,
    get_nearest_polynomial,
    get_osv_list_from_orbit,
    get_path_aux_cal,
    get_start_end_track,
    get_swath_misc_metadata,
    is_eap_correction_necessary,
    parse_polynomial_element,
    _get_manifest_pattern,
)


def _get_stripmap_center_and_border(tree):
    """Extract center point and border polygon for the full stripmap swath.

    Uses the first and last line of the geolocation grid (unlike IW where
    each pair of consecutive lines defines one burst boundary).

    Returns
    -------
    center : shapely.geometry.Point
    border : list[shapely.geometry.Polygon]
    """
    grid_pt_list = tree.find("geolocationGrid/geolocationGridPointList")
    n_grid_pts = int(grid_pt_list.attrib["count"])

    lines = np.empty(n_grid_pts)
    lats  = np.empty(n_grid_pts)
    lons  = np.empty(n_grid_pts)
    for i, gp in enumerate(grid_pt_list):
        lines[i] = int(gp[2].text)
        lats[i]  = float(gp[4].text)
        lons[i]  = float(gp[5].text)

    unique_lines = np.unique(lines)
    mask0 = lines == unique_lines[0]
    mask1 = lines == unique_lines[-1]

    swath_lons = np.concatenate((lons[mask0], lons[mask1][::-1]))
    swath_lats = np.concatenate((lats[mask0], lats[mask1][::-1]))

    center = calculate_centroid(swath_lons, swath_lats)
    poly   = shapely.geometry.Polygon(zip(swath_lons, swath_lats))
    border = check_dateline(poly)

    return center, border


def _ascending_node_time_from_manifest(tree_manifest):
    """Extract ascending node time from manifest.safe."""
    search_term, nsmap = _get_manifest_pattern(
        tree_manifest, ["orbitReference", "ascendingNodeTime"]
    )
    el = tree_manifest.find(search_term, nsmap)
    if el is not None:
        return as_datetime(el.text)
    return None


def stripmap_from_xml(
    annotation_path: str,
    orbit_path: str,
    tiff_path: str,
    open_method=open,
    flag_apply_eap: bool = True,
) -> list[Sentinel1BurstSlc]:
    """Parse a Sentinel-1 Stripmap SLC annotation XML.

    Parameters
    ----------
    annotation_path : str
        Path to the stripmap annotation XML (inside the SAFE structure).
    orbit_path : str
        Path to POEORB/RESORB .EOF file.
    tiff_path : str
        Path (or /vsizip/... path) to the measurement TIFF.
    open_method : callable
        File-open function; use zipfile.ZipFile.open when reading from zip.
    flag_apply_eap : bool
        Apply EAP phase correction if required by IPF version.

    Returns
    -------
    list[Sentinel1BurstSlc]
        Single-element list — one object for the full stripmap swath.
    """
    _, tail = os.path.split(annotation_path)
    # annotation filename: s1c-s3-slc-vv-<dates>-<orbit>-<uid>-<num>.xml
    parts       = tail.split("-")
    platform_id = parts[0].upper()          # S1A / S1B / S1C / S1D
    mode        = parts[1].upper()          # S1 / S2 / S3 / S4 / S5 / S6
    pol         = parts[3].upper()          # VV / VH / HH / HV
    safe_filename = os.path.basename(annotation_path.split(".SAFE")[0])

    # ── manifest.safe ─────────────────────────────────────────────────────────
    manifest_path = (
        os.path.dirname(annotation_path).replace("annotation", "") + "manifest.safe"
    )
    with (
        open_method(manifest_path, "r") as f_manifest,
        open_method(annotation_path, "r") as f_lads,
    ):
        tree_manifest = ET.parse(f_manifest)
        ipf_version   = get_ipf_version(tree_manifest)
        start_track, end_track = get_start_end_track(tree_manifest)
        anx_time_manifest = _ascending_node_time_from_manifest(tree_manifest)

        tree_lads      = ET.parse(f_lads)
        product_annotation = ProductAnnotation.from_et(tree_lads)

        # RFI
        rfi_annotation_path = annotation_path.replace(
            "annotation/", "annotation/rfi/rfi-"
        )
        try:
            with open_method(rfi_annotation_path, "r") as f_rads:
                tree_rads = ET.parse(f_rads)
                burst_rfi_info_swath = SwathRfiInfo.from_et(
                    tree_rads, tree_lads, ipf_version
                )
        except (FileNotFoundError, KeyError):
            if ipf_version >= RFI_INFO_AVAILABLE_FROM:
                warnings.warn(
                    f"RFI annotation expected (IPF={ipf_version} >= "
                    f"{RFI_INFO_AVAILABLE_FROM}) but not loaded."
                )
            burst_rfi_info_swath = None

        swath_misc_metadata = get_swath_misc_metadata(
            tree_manifest, tree_lads, product_annotation
        )

    # ── calibration ───────────────────────────────────────────────────────────
    try:
        cal_path = annotation_path.replace(
            "annotation/", "annotation/calibration/calibration-"
        )
        with open_method(cal_path, "r") as f_cads:
            calibration_annotation = CalibrationAnnotation.from_et(
                ET.parse(f_cads), cal_path
            )
    except (FileNotFoundError, KeyError):
        calibration_annotation = None

    # ── noise ─────────────────────────────────────────────────────────────────
    # Stripmap noise annotations have an empty noiseAzimuthVectorList, which
    # can cause NoiseAnnotation.from_et to fail. Treat any parsing error as
    # "no noise annotation".
    try:
        noise_path = annotation_path.replace(
            "annotation/", "annotation/calibration/noise-"
        )
        with open_method(noise_path, "r") as f_nads:
            noise_annotation = NoiseAnnotation.from_et(
                ET.parse(f_nads), ipf_version, noise_path
            )
    except (FileNotFoundError, KeyError, IndexError):
        noise_annotation = None

    # ── AUX_CAL / EAP ─────────────────────────────────────────────────────────
    eap_necessity = is_eap_correction_necessary(ipf_version)
    if eap_necessity.phase_correction and flag_apply_eap:
        path_aux_cals = os.path.join(
            f"{os.path.dirname(s1_annotation.__file__)}", "data", "aux_cal"
        )
        path_aux_cal  = get_path_aux_cal(path_aux_cals, annotation_path)
        if path_aux_cal is None:
            raise FileNotFoundError(
                f"Cannot find AUX_CAL in {path_aux_cals} for {platform_id}."
            )
        subswath_id    = tail.split("-")[1]       # e.g. "s3"
        aux_cal_subswath = AuxCal.load_from_zip_file(
            path_aux_cal, pol.lower(), subswath_id
        )
    else:
        aux_cal_subswath = None

    # ── main annotation parse ─────────────────────────────────────────────────
    with open_method(annotation_path, "r") as f:
        tree = ET.parse(f)

        prod_info = tree.find("generalAnnotation/productInformation")
        # Stripmap has no beam steering
        azimuth_steer_rate  = 0.0
        radar_freq          = float(prod_info.find("radarFrequency").text)
        range_sampling_rate = float(prod_info.find("rangeSamplingRate").text)
        orbit_direction     = prod_info.find("pass").text

        img_info = tree.find("imageAnnotation/imageInformation")
        average_azimuth_pixel_spacing = float(img_info.find("azimuthPixelSpacing").text)
        azimuth_time_interval         = float(img_info.find("azimuthTimeInterval").text)
        slant_range_time              = float(img_info.find("slantRangeTime").text)
        ascending_node_time_annotation = as_datetime(
            img_info.find("ascendingNodeTime").text
        )
        sensing_start = as_datetime(img_info.find("productFirstLineUtcTime").text)
        sensing_stop  = as_datetime(img_info.find("productLastLineUtcTime").text)

        # Stripmap: linesPerBurst=0 / samplesPerBurst=0 → use imageInformation
        n_lines   = int(img_info.find("numberOfLines").text)
        n_samples = int(img_info.find("numberOfSamples").text)

        downlink_el = tree.find(
            "generalAnnotation/downlinkInformationList/downlinkInformation"
        )
        prf_raw_data       = float(downlink_el.find("prf").text)
        rank               = int(downlink_el.find("downlinkValues/rank").text)
        range_chirp_rate   = float(
            downlink_el.find("downlinkValues/txPulseRampRate").text
        )

        orbit_number = int(tree.find("adsHeader/absoluteOrbitNumber").text)

        # polynomials
        az_fm_rate_list = [
            parse_polynomial_element(x, "azimuthFmRatePolynomial")
            for x in tree.find("generalAnnotation/azimuthFmRateList")
        ]
        doppler_list = [
            parse_polynomial_element(x, "dataDcPolynomial")
            for x in tree.find("dopplerCentroid/dcEstimateList")
        ]

        rng_proc = tree.find(
            "imageAnnotation/processingInformation/"
            "swathProcParamsList/swathProcParams/rangeProcessing"
        )
        rng_processing_bandwidth = float(rng_proc.find("processingBandwidth").text)
        range_window_type        = str(rng_proc.find("windowType").text)
        range_window_coeff       = float(rng_proc.find("windowCoefficient").text)

    # ── derived quantities ────────────────────────────────────────────────────
    wavelength        = isce3.core.speed_of_light / radar_freq
    starting_range    = slant_range_time * isce3.core.speed_of_light / 2
    range_pxl_spacing = isce3.core.speed_of_light / (2 * range_sampling_rate)

    # Mid-swath range (replaces iw2_mid_range for stripmap)
    sm_mid_range = starting_range + 0.5 * n_samples * range_pxl_spacing

    # ── orbit ─────────────────────────────────────────────────────────────────
    if orbit_path:
        osv_list = get_osv_list_from_orbit(orbit_path, sensing_start, sensing_stop)
        orbit    = get_burst_orbit(sensing_start, sensing_stop, osv_list)

        # Use ANX from orbit state vectors; fall back to annotation value
        try:
            from s1reader.s1_reader import get_ascending_node_time_orbit
            ascending_node_time = get_ascending_node_time_orbit(
                osv_list, sensing_start, ascending_node_time_annotation
            )
        except (ValueError, ImportError):
            ascending_node_time = (
                anx_time_manifest or ascending_node_time_annotation
            )
    else:
        warnings.warn("No orbit file provided; using annotation ascending node time.")
        osv_list            = []
        orbit               = None
        ascending_node_time = anx_time_manifest or ascending_node_time_annotation

    # ── burst ID — use mid-swath sensing time ─────────────────────────────────
    # Stripmap has no ESA burst delineation; derive a stable ID from the
    # sensing time using the same S1BurstId machinery as IW TOPS.
    mid_sensing_time = sensing_start + datetime.timedelta(
        seconds=0.5 * n_lines * azimuth_time_interval
    )
    burst_id = S1BurstId.from_burst_params(
        mid_sensing_time,
        ascending_node_time,
        start_track,
        end_track,
        mode,           # e.g. "S3" — used as the subswath label
    )

    # ── select polynomials at mid-swath time ─────────────────────────────────
    az_fm_rate = get_nearest_polynomial(mid_sensing_time, az_fm_rate_list)
    poly1d     = get_nearest_polynomial(mid_sensing_time, doppler_list)
    lut2d      = doppler_poly1d_to_lut2d(
        poly1d, starting_range, range_pxl_spacing,
        (n_lines, n_samples), azimuth_time_interval
    )
    doppler = Doppler(poly1d, lut2d)

    # ── geometry ──────────────────────────────────────────────────────────────
    with open_method(annotation_path, "r") as f:
        tree_geo = ET.parse(f)
    center, border = _get_stripmap_center_and_border(tree_geo)

    # ── valid pixel bounds — full swath (no per-burst mask in stripmap) ───────
    first_valid_sample = 0
    last_valid_sample  = n_samples - 1
    first_valid_line   = 0
    last_valid_line    = n_lines   - 1

    # ── calibration / noise / EAP per swath ──────────────────────────────────
    if calibration_annotation is None:
        # Stub: compass h5_helpers accesses basename_cads unconditionally
        burst_calibration = BurstCalibration(basename_cads="stripmap")
    else:
        burst_calibration = BurstCalibration.from_calibration_annotation(
            calibration_annotation, sensing_start
        )

    if noise_annotation is None:
        # Stub: compass h5_helpers accesses basename_nads unconditionally
        ones = np.ones(2)
        burst_noise = BurstNoise(
            basename_nads="stripmap",
            range_azimuth_time=sensing_start,
            range_line=0.0,
            range_pixel=np.array([0.0, float(n_samples - 1)]),
            range_lut=ones,
            azimuth_first_azimuth_line=0,
            azimuth_first_range_sample=0,
            azimuth_last_azimuth_line=n_lines - 1,
            azimuth_last_range_sample=n_samples - 1,
            azimuth_line=np.array([0.0, float(n_lines - 1)]),
            azimuth_lut=ones,
            line_from=0,
            line_to=n_lines - 1,
        )
    else:
        burst_noise = BurstNoise.from_noise_annotation(
            noise_annotation,
            sensing_start,
            0,
            n_lines - 1,
            ipf_version,
        )

    if aux_cal_subswath is None:
        burst_eap = None
    else:
        burst_eap = BurstEAP.from_product_annotation_and_aux_cal(
            product_annotation, aux_cal_subswath, sensing_start
        )

    # ── extended FM/DC coefficients ───────────────────────────────────────────
    extended_coeffs = BurstExtendedCoeffs.from_polynomial_lists(
        az_fm_rate_list,
        doppler_list,
        sensing_start,
        sensing_stop,
    )

    # ── RFI ──────────────────────────────────────────────────────────────────
    if burst_rfi_info_swath is None:
        burst_rfi_info = None
    else:
        burst_rfi_info = burst_rfi_info_swath.extract_by_aztime(sensing_start)

    # ── misc metadata ─────────────────────────────────────────────────────────
    burst_misc_metadata = swath_misc_metadata.extract_by_aztime(sensing_start)

    slc = Sentinel1BurstSlc(
        ipf_version,
        sensing_start,
        radar_freq,
        wavelength,
        azimuth_steer_rate,
        average_azimuth_pixel_spacing,
        azimuth_time_interval,
        slant_range_time,
        starting_range,
        sm_mid_range,           # iw2_mid_range field — set to SM mid range
        range_sampling_rate,
        range_pxl_spacing,
        (n_lines, n_samples),
        az_fm_rate,
        doppler,
        rng_processing_bandwidth,
        pol,
        burst_id,
        platform_id,
        safe_filename,
        center,
        border,
        orbit,
        orbit_direction,
        orbit_number,
        tiff_path,
        0,                      # i_burst — single swath
        first_valid_sample,
        last_valid_sample,
        first_valid_line,
        last_valid_line,
        range_window_type,
        range_window_coeff,
        rank,
        prf_raw_data,
        range_chirp_rate,
        burst_calibration,
        burst_noise,
        burst_eap,
        extended_coeffs,
        burst_rfi_info,
        burst_misc_metadata,
    )

    return [slc]


def load_stripmap_burst(
    path: str,
    orbit_path: str,
    pol: str = "vv",
    flag_apply_eap: bool = True,
) -> list[Sentinel1BurstSlc]:
    """Load Sentinel-1 Stripmap SLC from a SAFE zip or directory.

    Parameters
    ----------
    path : str
        Path to SAFE zip or SAFE directory.
    orbit_path : str
        Path to POEORB/RESORB .EOF file.
    pol : str
        Polarization: "vv", "vh", "hh", "hv".
    flag_apply_eap : bool
        Apply EAP phase correction if required by IPF version.

    Returns
    -------
    list[Sentinel1BurstSlc]
        Single-element list containing the stripmap swath burst object.
    """
    pol = pol.lower()
    if pol not in ("vv", "vh", "hh", "hv"):
        raise ValueError(f"Unknown polarization: {pol}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    if os.path.isdir(path):
        return _stripmap_from_safe_dir(path, pol, orbit_path, flag_apply_eap)
    elif os.path.isfile(path):
        return _stripmap_from_zip(path, pol, orbit_path, flag_apply_eap)
    else:
        raise ValueError(f"Unsupported path: {path}")


def _stripmap_from_zip(
    zip_path: str, pol: str, orbit_path: str, flag_apply_eap: bool
) -> list[Sentinel1BurstSlc]:
    with zipfile.ZipFile(zip_path, "r") as z_file:
        z_names = z_file.namelist()

        # annotation: annotation/s1?-s?-slc-{pol}-*.xml  (not in calibration/ or rfi/)
        def _is_sm_annotation(p):
            toks = p.split("/")
            return (
                toks[-2] == "annotation"
                and f"-slc-{pol}-" in toks[-1]
                and toks[-1].endswith(".xml")
            )

        f_ann = [n for n in z_names if _is_sm_annotation(n)]
        if not f_ann:
            raise ValueError(f"No {pol} annotation found in {zip_path}")
        f_ann = f_ann[0]

        # tiff
        f_tiff = [
            n for n in z_names
            if "measurement" in n and f"-slc-{pol}-" in n and n.endswith(".tiff")
        ]
        f_tiff = f"/vsizip/{zip_path}/{f_tiff[0]}" if f_tiff else ""

        return stripmap_from_xml(
            f_ann, orbit_path, f_tiff, z_file.open, flag_apply_eap
        )


def _stripmap_from_safe_dir(
    safe_dir: str, pol: str, orbit_path: str, flag_apply_eap: bool
) -> list[Sentinel1BurstSlc]:
    ann_dir = os.path.join(safe_dir, "annotation")
    f_ann = [
        f for f in os.listdir(ann_dir)
        if f"-slc-{pol}-" in f and f.endswith(".xml")
    ]
    if not f_ann:
        raise ValueError(f"No {pol} annotation found in {safe_dir}")
    f_ann = os.path.join(ann_dir, f_ann[0])

    meas_dir = os.path.join(safe_dir, "measurement")
    if os.path.isdir(meas_dir):
        import glob as _glob
        f_tiff = _glob.glob(os.path.join(meas_dir, f"*-slc-{pol}-*.tiff"))
        f_tiff = f_tiff[0] if f_tiff else ""
    else:
        warnings.warn(f"measurement/ not found in {safe_dir}")
        f_tiff = ""

    return stripmap_from_xml(f_ann, orbit_path, f_tiff, flag_apply_eap=flag_apply_eap)
