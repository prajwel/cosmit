#!/usr/bin/env python3
"""Cosmit: Cosmic-ray tools for AstroSat/UVIT."""

import csv
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from astropy.io import fits
from datetime import datetime, timezone


__version__ = "0.1.0"
DEFAULT_THRESHOLD_FACTOR = 10.0
WARN_FLAGGED_FRAME_PERCENT = 10.0


class CosmitError(RuntimeError):
    """Raised when Cosmit cannot process an event list."""


class CosmitPaths:
    """Paths for the products generated from one UVIT event list."""

    def __init__(self, events_list, output_dir):
        self.events_list = Path(events_list).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.product_id = self.events_list.name.replace("_l2ce.fits", "")
        self.diagnostics_dir = self.output_dir / "diagnostics" / self.product_id
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_events_list = self.output_dir / self.events_list.name

    @property
    def log_file(self):
        return self.diagnostics_dir / "cosmit.log"

    @property
    def unique_frametime_counts_plot(self):
        return self.diagnostics_dir / "unique_frametime_counts.png"

    @property
    def unique_frametime_counts_time_plot(self):
        return self.diagnostics_dir / "unique_frametime_counts_vs_time.png"

    @property
    def cosmic_ray_catalogue_csv(self):
        return self.diagnostics_dir / "cosmic_ray_catalogue.csv"

    @property
    def detector_image(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "I_l2img.fits"
        )

    @property
    def detector_error(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "I_l2err.fits"
        )

    @property
    def astronomical_image(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "A_l2img.fits"
        )

    @property
    def astronomical_error(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "A_l2err.fits"
        )

    @property
    def detector_exposure(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "I_l2exp.fits"
        )

    @property
    def astronomical_exposure(self):
        return self.output_dir / self.events_list.name.replace(
            "_l2ce.fits", "A_l2exp.fits"
        )


def _utc_now():
    return (
        datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        + " UTC"
    )


def _read_events_list(events_hdu):
    table = events_hdu[1].data
    time = np.asarray(table["MJD_L2"])
    photons = np.asarray(table["EFFECTIVE_NUM_PHOTONS"])
    bad_flag = np.asarray(table["BAD FLAG"])
    good = (photons > 0) & bad_flag.astype(bool)
    unique_frametimes, unique_frametime_counts = np.unique(
        time[good], return_counts=True
    )
    return unique_frametimes, unique_frametime_counts


def _write_log(paths, input_source, summary):
    percentage_keys = {
        "Flagged frames fraction",
        "Flagged valid events fraction",
        "Removed events fraction",
    }
    precision = {
        "Median events/frame": 3,
        "NMAD events/frame": 3,
        "Threshold (events/frame; reject if count > threshold)": 4,
    }

    with open(paths.log_file, "w", encoding="utf-8") as logfile:
        logfile.write("Product information\n")
        logfile.write("-------------------\n")
        logfile.write(f"Product ID        : {paths.product_id}\n")
        logfile.write(f"Input source      : {input_source}\n")
        logfile.write(f"Input file        : {paths.events_list}\n")
        logfile.write(f"Output file       : {paths.cleaned_events_list}\n")
        logfile.write("\nRun information\n")
        logfile.write("---------------\n")
        logfile.write(f"Cosmit version    : {__version__}\n")
        logfile.write(f"Run time          : {_utc_now()}\n")
        logfile.write("\nAlgorithm parameters and results\n")
        logfile.write("--------------------------------\n")

        for key, value in summary.items():
            if key in percentage_keys:
                value = f"{value:.4f}%"
            elif key in precision:
                value = f"{value:.{precision[key]}f}"
            logfile.write(f"{key:<65}: {value}\n")


def _write_cosmic_ray_catalogue(
    filename, unique_frametimes, unique_frametime_counts, flags
):
    np.savetxt(
        filename,
        np.column_stack((unique_frametimes, unique_frametime_counts, flags)),
        delimiter=",",
        fmt=["%.9f", "%d", "%s"],
        header="MJD_L2,event_count,CR_flag",
        comments="",
    )


def _plot_frame_diagnostics(
    paths, unique_frametimes, unique_frametime_counts, flagged, threshold
):
    # Histogram: logarithmic axes make the long high-count tail visible.
    positive = unique_frametime_counts[unique_frametime_counts > 0]
    max_count = int(positive.max())
    upper = max(10, int(np.ceil(np.log10(max_count + 1))))
    bins = np.logspace(0, upper, 90)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(positive, bins=bins)
    ax.axvline(
        threshold, color="tab:red", linestyle="--", label=f"Threshold = {threshold:.1f}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Events per frame")
    ax.set_ylabel("Number of frames")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths.unique_frametime_counts_plot, dpi=150)
    plt.close(fig)

    # Limit the background points for very large event lists; always show all flagged frames.
    stride = max(1, len(unique_frametime_counts) // 100_000)
    sample = np.arange(0, len(unique_frametime_counts), stride)
    relative_time = unique_frametimes - unique_frametimes[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(
        relative_time[sample],
        unique_frametime_counts[sample],
        s=2,
        alpha=0.35,
        label="Frames",
    )
    if np.any(flagged):
        ax.scatter(
            relative_time[flagged],
            unique_frametime_counts[flagged],
            s=2,
            color="tab:red",
            alpha=0.8,
            label="Flagged frames",
        )
    ax.axhline(
        threshold, color="tab:red", linestyle="--", label=f"Threshold = {threshold:.1f}"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time relative to first retained frame (s)")
    ax.set_ylabel("Events per frame")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths.unique_frametime_counts_time_plot, dpi=150)
    plt.close(fig)


def _add_cosmit_header(header, summary, input_source):
    header["COSMIT"] = (True, "Processed using Cosmit")
    header["COS_VER"] = (__version__, "Cosmit version")
    header["COS_SRC"] = (input_source, "Input product source")

    header["COS_FACT"] = (float(summary["threshold factor"]), "NMAD threshold factor")
    header["COS_MED"] = (float(summary["Median events/frame"]), "Median events per frame")
    header["COS_NMAD"] = (float(summary["NMAD events/frame"]), "Normal-scaled median absolute deviation")
    header["COS_THR"] = (
        float(summary["Threshold (events/frame; reject if count > threshold)"]),
        "Frame rejection threshold",
    )

    header["COS_TFR"] = (int(summary["Total frames"]), "Total frames in events list")
    header["COS_FFR"] = (int(summary["Flagged frames"]), "Frames flagged by Cosmit")
    header["COS_FFRF"] = (float(summary["Flagged frames fraction"]), "Percentage of frames flagged by Cosmit")
    header["COS_EFAC"] = (
    float(summary["Exposure scale factor"]),
    "Fraction of exposure retained")

    header["COS_TVE"] = (int(summary["Total valid events"]), "Total valid events")
    header["COS_FVE"] = (int(summary["Flagged valid events"]), "Valid events in flagged frames")
    header["COS_FVEF"] = (float(summary["Flagged valid events fraction"]), "Percentage of valid events in flagged frames")

    header["COS_TIE"] = (int(summary["Total input events"]), "Total input events")
    header["COS_REV"] = (int(summary["Removed events"]), "Events removed by Cosmit")
    header["COS_REVF"] = (float(summary["Removed events fraction"]), "Percentage of input events removed")

    header.insert("COSMIT", ("", ""))
    header.insert("COSMIT", ("", "Cosmit processing information"))
    header.insert("COSMIT", ("", ""))


def generate_image_products(
    events_list, exposure_map, image_file, error_file, astronomical=False
):
    """Generate count-rate and error images"""
    x_key, y_key = (
        ("Fx_astronomical", "Fy_astronomical") if astronomical else ("Fx", "Fy")
    )
    with fits.open(events_list, memmap=False) as events_hdu, fits.open(
        exposure_map
    ) as exp_hdu:
        event_header = events_hdu[0].header
        table = events_hdu[1].data

        fx = np.asarray(table[x_key])
        fy = np.asarray(table[y_key])
        photons = np.asarray(table["EFFECTIVE_NUM_PHOTONS"])
        bad_flag = np.asarray(table["BAD FLAG"]).astype(bool)

        mask = (photons > 0) & bad_flag
        fx, fy, photons = fx[mask], fy[mask], photons[mask]
        edges = np.arange(-0.5, 4800.5, 1)
        events, _, _ = np.histogram2d(fy, fx, bins=(edges, edges), weights=photons)
        counts, _, _ = np.histogram2d(fy, fx, bins=(edges, edges))
        exposure = np.asarray(exp_hdu[0].data, dtype=float)
        denominator = exposure * float(event_header["AVGFRMRT"])
        with np.errstate(divide="ignore", invalid="ignore"):
            cps = events / denominator
            cps_error = cps / np.sqrt(counts)
        cps[exposure == 0] = 0
        cps_error[(exposure == 0) | (counts == 0)] = 0
        cps = cps.astype(np.float32)
        cps_error = cps_error.astype(np.float32)

        image_header = exp_hdu[0].header.copy()

        image_header["FILENAME"] = image_file.name
        image_header["FILEDATE"] = _utc_now().replace(" UTC", "")
        image_header["FILEORIG"] = "Cosmit"
        image_header["DATATYPE"] = "count-rate image"
        fits.PrimaryHDU(cps, header=image_header).writeto(image_file, overwrite=True)

        error_header = image_header.copy()
        error_header["FILENAME"] = error_file.name
        error_header["DATATYPE"] = "count-rate error image"
        fits.PrimaryHDU(cps_error, header=error_header).writeto(
            error_file, overwrite=True
        )


def _scale_exposure_map(input_map, output_map, scale_factor, summary, input_source):
    with fits.open(input_map, memmap=False) as hdul:
        hdul[0].data = hdul[0].data.astype(np.float64) * scale_factor
        hdul[0].data = hdul[0].data.astype(np.float32)
        _add_cosmit_header(hdul[0].header, summary, input_source)
        hdul[0].header["FILENAME"] = Path(output_map).name
        hdul[0].header["FILEDATE"] = _utc_now().replace(" UTC", "")
        hdul[0].header["FILEORIG"] = "Cosmit"
        hdul.writeto(output_map, overwrite=True)

def generate_products(paths, input_source_dir, summary, input_source):
    exposure_scale_factor = summary["Exposure scale factor"]

    input_source_dir = Path(input_source_dir)
    input_detector_exposure = input_source_dir / paths.detector_exposure.name
    input_astronomical_exposure = input_source_dir / paths.astronomical_exposure.name

    for source, destination in (
        (input_detector_exposure, paths.detector_exposure),
        (input_astronomical_exposure, paths.astronomical_exposure),
    ):
        _scale_exposure_map(source, destination, exposure_scale_factor, summary, input_source)

    generate_image_products(
        paths.cleaned_events_list,
        paths.detector_exposure,
        paths.detector_image,
        paths.detector_error,
    )

    generate_image_products(
        paths.cleaned_events_list,
        paths.astronomical_exposure,
        paths.astronomical_image,
        paths.astronomical_error,
        astronomical=True,
    )


def process_events_list(events_list, output_dir, threshold_factor=DEFAULT_THRESHOLD_FACTOR, input_source="v7"):
    """Remove frames having a total number of events above median + threshold_factor * NMAD."""
    paths = CosmitPaths(events_list, output_dir)

    with fits.open(paths.events_list, memmap=False) as input_hdul:
        unique_frametimes, unique_frametime_counts = _read_events_list(input_hdul)
        median = float(np.median(unique_frametime_counts))
        mad_std = float(stats.median_abs_deviation(unique_frametime_counts, scale="normal"))
        if mad_std == 0:
            raise CosmitError(
                "The frame-count NMAD is zero, so median + threshold_factor * NMAD cannot provide a useful threshold."
            )

        threshold = median + float(threshold_factor) * mad_std
        flagged = unique_frametime_counts > threshold
        flagged_times = unique_frametimes[flagged]
        original_table = input_hdul[1].data
        all_times = np.asarray(original_table["MJD_L2"])
        rows_to_keep = ~np.isin(all_times, flagged_times)


        old_table_hdu = input_hdul[1]
        clean_table_hdu = fits.BinTableHDU.from_columns(
            old_table_hdu.columns,
            nrows=int(np.count_nonzero(rows_to_keep)),
            header=old_table_hdu.header.copy(),
            name=old_table_hdu.name,
        )
        for column_name in old_table_hdu.columns.names:
            clean_table_hdu.data[column_name] = original_table[column_name][
                rows_to_keep
            ]
        clean_hdul = fits.HDUList(
            [
                clean_table_hdu if index == 1 else hdu.copy()
                for index, hdu in enumerate(input_hdul)
            ]
        )

        frames_inspected = len(unique_frametime_counts)
        frames_flagged = int(np.count_nonzero(flagged))
        frames_flagged_fraction = 100 * np.mean(flagged)
        exposure_scale_factor = 1 - (frames_flagged / frames_inspected)
        
        total_valid_events = int(unique_frametime_counts.sum())
        flagged_valid_events = int(unique_frametime_counts[flagged].sum())
        flagged_valid_events_fraction = flagged_valid_events * 100 / total_valid_events

        
        total_input_events = len(original_table)
        removed_events = int(np.count_nonzero(~rows_to_keep))
        removed_events_fraction = removed_events * 100 / total_input_events
        
        summary = {
            "threshold factor": threshold_factor,
            "Median events/frame": median,
            "NMAD events/frame": mad_std,
            "Threshold (events/frame; reject if count > threshold)": threshold,
            "Total frames": frames_inspected,
            "Flagged frames": frames_flagged,
            "Flagged frames fraction": frames_flagged_fraction,
            "Exposure scale factor": exposure_scale_factor,
            "Total valid events": total_valid_events,
            "Flagged valid events": flagged_valid_events,
            "Flagged valid events fraction": flagged_valid_events_fraction,
            "Total input events": total_input_events,
            "Removed events": removed_events,
            "Removed events fraction": removed_events_fraction,
        }

        _add_cosmit_header(clean_hdul[0].header, summary, input_source)
        clean_hdul.writeto(paths.cleaned_events_list, overwrite=True, checksum=True)
        clean_hdul.close()

    _write_cosmic_ray_catalogue(
        paths.cosmic_ray_catalogue_csv,
        unique_frametimes,
        unique_frametime_counts,
        flagged,
    )
    _plot_frame_diagnostics(
        paths, unique_frametimes, unique_frametime_counts, flagged, threshold
    )
    _write_log(paths, input_source, summary)

    flagged_percent = 100 * np.mean(flagged)
    if flagged_percent > WARN_FLAGGED_FRAME_PERCENT:
        warnings.warn(
            f"{paths.product_id}: {flagged_percent:.3f}% of frames were flagged.",
            UserWarning,
            stacklevel=2,
        )

    return paths, summary


def _output_directory_for(events_list, source_marker):
    events_list = Path(events_list).resolve()
    source_root = next(
        (parent for parent in events_list.parents if parent.name == source_marker), None
    )
    if source_root is None:
        raise CosmitError(
            f"Could not locate a {source_marker!r} directory above {events_list}."
        )
    relative_parent = events_list.parent.relative_to(source_root)
    return source_root.parent / "cosmit_data_products" / relative_parent


def process_observation(observation_dir, input_source="v7", threshold_factor=DEFAULT_THRESHOLD_FACTOR):
    f"""Process event lists in an observation from v7 or Resolvit products.

    Parameters
    ----------
    observation_dir : str or pathlib.Path
        Root directory of the UVIT observation.

    input_source : {'v7', 'resolvit'}, default='v7'
        Select default Level2 pipeline v7 products under ``data_products`` or Resolvit
        products under ``resolvit_data_products``. Results are always written to
        a sibling ``cosmit_data_products`` directory.

    threshold_factor : float, default={DEFAULT_THRESHOLD_FACTOR}
        Threshold multiplier in median + threshold_factor * NMAD.
    """
    source_dirs = {"v7": "data_products", "resolvit": "resolvit_data_products"}
    if input_source not in source_dirs:
        raise ValueError("input_source must be 'v7' or 'resolvit'.")

    observation_dir = Path(observation_dir).resolve()
    source_marker = source_dirs[input_source]
    events_lists = []
    for path in observation_dir.rglob("AS1*l2ce.fits"):
        if source_marker in path.parts:
            events_lists.append(path)
    events_lists = sorted(set(events_lists))
    if not events_lists:
        raise FileNotFoundError(
            f"No AS1*l2ce.fits event lists found under {observation_dir} for source {input_source!r}."
        )

    results = []
    output_paths = set()
    for events_list in events_lists:
        output_dir = _output_directory_for(events_list, source_marker)
        output_key = (output_dir.resolve(), events_list.name)
        if output_key in output_paths:
            raise CosmitError(
                f"Multiple input event lists map to the same Cosmit output: {output_dir / events_list.name}"
            )
        output_paths.add(output_key)

        print(f"Processing {events_list.name} [{input_source}]")
        paths, summary = process_events_list(
            events_list, output_dir, threshold_factor=threshold_factor, input_source=input_source
        )
        generate_products(paths, events_list.parent, summary, input_source)
        results.append((paths, summary))

    return results
