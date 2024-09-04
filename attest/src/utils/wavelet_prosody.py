# ATTEST: an Analytics Tool for the Testing and Evaluation of Speech Technologies
#
# Copyright (C) 2024 Constructor Technology AG
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see: <http://www.gnu.org/licenses/>.
#

# System
import os

# Configuration
import yaml
from collections import defaultdict

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt

# acoustic features
from wavelet_prosody_toolkit.prosody_tools import energy_processing
from wavelet_prosody_toolkit.prosody_tools import f0_processing
from wavelet_prosody_toolkit.prosody_tools import duration_processing

# helpers
from wavelet_prosody_toolkit.prosody_tools import misc
from wavelet_prosody_toolkit.prosody_tools import smooth_and_interp

# wavelet transform
from wavelet_prosody_toolkit.prosody_tools import cwt_utils, loma, lab

# project related
from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.logger import get_logger


_wavelet_prosody_extractor = None


def get_wavelet_prosody_extractor():
    global _wavelet_prosody_extractor
    if _wavelet_prosody_extractor is None:
        _wavelet_prosody_extractor = WaveletProsodyExtractor()

    return _wavelet_prosody_extractor


settings = get_settings()


class WaveletProsodyExtractor:

    def __init__(self):
        self.logger = get_logger()
        self.cache_dir = settings.CACHE_DIR
        self.cache_filename = "wavelet_prosody_features.pickle"
        self.config_path = "attest/third_party/wavelet_prosody_toolkit/configs/default.yaml"

    def extract_features_for_project(self, project: Project, word_alignments: list):
        features = []

        self.logger.info("Creating wavelet prosody plots...")

        configuration = defaultdict()
        with open(self.config_path, "r") as f:
            configuration = _apply_configuration(
                configuration,
                defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)),
            )

        for i, (audio_file, word_ali) in enumerate(zip(project.audio_files, word_alignments)):
            output_file = os.path.join(self.cache_dir, project.name, "wavelet_prosody", f"image_{i}.png")
            if not os.path.exists(output_file):
                self.logger.info(f"Creating wavelet prosody plots for {audio_file}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                _ = _analysis(audio_file, word_ali, configuration, output_file)
            features.append({"image": output_file})

        self.logger.info("Creating wavelet prosody plots is done!")

        return features


def _apply_configuration(current_configuration, updating_part):
    """Utils to update the current configuration using the updating part

    Parameters
    ----------
    current_configuration: dict
        The current state of the configuration

    updating_part: dict
        The information to add to the current configuration

    Returns
    -------
    dict
    the updated configuration
    """
    if not isinstance(current_configuration, dict):
        return updating_part

    if current_configuration is None:
        return updating_part

    if updating_part is None:
        return current_configuration

    for k in updating_part:
        if k not in current_configuration:
            current_configuration[k] = updating_part[k]
        else:
            current_configuration[k] = _apply_configuration(current_configuration[k], updating_part[k])

    return current_configuration


def _analysis(input_file, word_ali, cfg, output_filename):

    # Load the wave file
    # print("Analyzing %s starting..." % input_file)
    orig_sr, sig = misc.read_wav(input_file)

    # extract energy
    energy = energy_processing.extract_energy(
        sig,
        orig_sr,
        cfg["energy"]["band_min"],
        cfg["energy"]["band_max"],
        cfg["energy"]["calculation_method"],
    )
    energy = np.cbrt(energy + 1)
    if cfg["energy"]["smooth_energy"]:
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)  # FIXME: 30? 3?
        energy = smooth_and_interp.smooth(energy, 10)

    # extract f0
    raw_pitch = f0_processing.extract_f0(
        sig,
        orig_sr,
        f0_min=cfg["f0"]["min_f0"],
        f0_max=cfg["f0"]["max_f0"],
        voicing=cfg["f0"]["voicing_threshold"],
        # harmonics=cfg["f0"]["harmonics"],
        configuration=cfg["f0"]["pitch_tracker"],
    )
    # interpolate, stylize
    pitch = f0_processing.process(raw_pitch)

    # extract speech rate
    rate = np.zeros(len(pitch))

    # Get annotations
    tiers = {"words": [(200 * start, 200 * end, word) for word, start, end in word_ali]}

    # Extract duration
    if len(tiers) > 0:
        dur_tiers = []
        for level in cfg["duration"]["duration_tiers"]:
            # TODO: may be for my case word-level tier is enough
            if not level.lower() in tiers:
                continue
            assert level.lower() in tiers, (
                level + " not defined in tiers: check that duration_tiers in config match the actual textgrid tiers"
            )
            try:
                dur_tiers.append(tiers[level.lower()])
            except KeyError:
                print("\nerror: " + '"' + level + '"' + " not in labels, modify duration_tiers in config\n\n")
                raise

    if not cfg["duration"]["acoustic_estimation"]:
        rate = duration_processing.get_duration_signal(
            dur_tiers,
            weights=cfg["duration"]["weights"],
            linear=cfg["duration"]["linear"],
            sil_symbols=cfg["duration"]["silence_symbols"],
            bump=cfg["duration"]["bump"],
        )

    else:
        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)

    if cfg["duration"]["delta_duration"]:
        rate = np.diff(rate)

    # Combine signals
    min_length = np.min([len(pitch), len(energy), len(rate)])
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    rate = rate[:min_length]

    if cfg["feature_combination"]["type"] == "product":
        pitch = misc.normalize_minmax(pitch) ** cfg["feature_combination"]["weights"]["f0"]
        energy = misc.normalize_minmax(energy) ** cfg["feature_combination"]["weights"]["energy"]
        rate = misc.normalize_minmax(rate) ** cfg["feature_combination"]["weights"]["duration"]
        params = pitch * energy * rate

    else:
        params = (
            misc.normalize_std(pitch) * cfg["feature_combination"]["weights"]["f0"]
            + misc.normalize_std(energy) * cfg["feature_combination"]["weights"]["energy"]
            + misc.normalize_std(rate) * cfg["feature_combination"]["weights"]["duration"]
        )

    if cfg["feature_combination"]["detrend"]:
        params = smooth_and_interp.remove_bias(params, 800)

    params = misc.normalize_std(params)

    # CWT analysis
    (cwt, scales, freqs) = cwt_utils.cwt_analysis(
        params,
        mother_name=cfg["wavelet"]["mother_wavelet"],
        period=cfg["wavelet"]["period"],
        num_scales=cfg["wavelet"]["num_scales"],
        scale_distance=cfg["wavelet"]["scale_distance"],
        apply_coi=False,
    )
    cwt = np.real(cwt)
    scales *= 200  # FIXME: why 200?

    # Compute lines of maximum amplitude
    assert cfg["labels"]["annotation_tier"].lower() in tiers, (
        cfg["labels"]["annotation_tier"]
        + " not defined in tiers: check that annotation_tier in config is found in the textgrid tiers"
    )
    labels = tiers[cfg["labels"]["annotation_tier"].lower()]

    # get scale corresponding to avg unit length of selected tier
    scale_dist = cfg["wavelet"]["scale_distance"]
    scales = (1.0 / freqs * 200) * 0.5  # FIXME: hardcoded vales
    unit_scale = misc.get_best_scale2(scales, labels)

    # Define the scale information (FIXME: description)
    pos_loma_start_scale = unit_scale + int(
        cfg["loma"]["prom_start"] / scale_dist
    )  # three octaves down from average unit length
    pos_loma_end_scale = unit_scale + int(cfg["loma"]["prom_end"] / scale_dist)
    neg_loma_start_scale = unit_scale + int(cfg["loma"]["boundary_start"] / scale_dist)  # two octaves down
    neg_loma_end_scale = unit_scale + int(cfg["loma"]["boundary_end"] / scale_dist)  # one octave up

    pos_loma = loma.get_loma(cwt, scales, pos_loma_start_scale, pos_loma_end_scale)
    neg_loma = loma.get_loma(-cwt, scales, neg_loma_start_scale, neg_loma_end_scale)

    max_loma = loma.get_prominences(pos_loma, labels)
    prominences = np.array(max_loma)
    boundaries = np.array(loma.get_boundaries(max_loma, neg_loma, labels))

    # Plotting
    fig, ax = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=(len(labels) / 10 * 8, 8),
        gridspec_kw={"height_ratios": [1, 1, 1, 2, 4, 1.5]},
    )
    plt.subplots_adjust(hspace=0)

    # Plot individual signals
    ax[0].plot(pitch, linewidth=1)
    ax[0].set_ylabel("Pitch", rotation="horizontal", ha="right", va="center")

    ax[1].plot(energy, linewidth=1)
    ax[1].set_ylabel("Energy", rotation="horizontal", ha="right", va="center")

    ax[2].plot(rate, linewidth=1)
    ax[2].set_ylabel("Speech rate", rotation="horizontal", ha="right", va="center")

    # Plot combined signal
    ax[3].plot(params, linewidth=1)
    ax[3].set_ylabel("Combined \n signal", rotation="horizontal", ha="right", va="center")
    plt.xlim(0, len(params))

    # Wavelet and loma
    cwt[cwt > 0] = np.log(cwt[cwt > 0] + 1.0)
    cwt[cwt < -0.1] = -0.1
    ax[4].contourf(cwt, 100, cmap="inferno")
    loma.plot_loma(pos_loma, ax[4], color="black")
    loma.plot_loma(neg_loma, ax[4], color="white")
    ax[4].set_ylabel("Wavelet & \n LOMA", rotation="horizontal", ha="right", va="center")

    # Add labels
    prom_text = prominences[:, 1] / (np.max(prominences[:, 1])) * 2.5 + 0.5
    lab.plot_labels(
        labels,
        ypos=0.3,
        size=6,
        prominences=prom_text,
        fig=ax[5],
        boundary=False,
        background=False,
    )
    ax[5].set_ylabel("Labels", rotation="horizontal", ha="right", va="center")
    for i in range(0, len(labels)):
        for a in [0, 1, 2, 3, 4, 5]:
            ax[a].axvline(x=labels[i][0], color="black", linestyle="-", linewidth=0.2, alpha=0.5)

            ax[a].axvline(
                x=labels[i][1],
                color="black",
                linestyle="-",
                linewidth=0.2 + boundaries[i][-1] * 2,
                alpha=0.5,
            )

    plt.xlim(0, cwt.shape[1])

    # Align ylabels and remove axis
    fig.align_ylabels(ax)
    for i in range(len(ax) - 1):
        ax[i].tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        ax[i].tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
        )  # labels along the bottom edge are off

    ax[len(ax) - 1].tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False,
    )  # labels along the bottom edge are off

    # Plot
    fig.savefig(output_filename, bbox_inches="tight", dpi=400)
    plt.close("all")
    plt.close()

    return {"image": output_filename}
