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

import yaml
from typing import Dict, List, Tuple
from pydantic import BaseModel, ValidationError

from attest.ui.model import MetricType


class Settings(BaseModel):
    LOGGING_LEVEL: str
    DATA_DIR: str
    MODELS_DIR: str
    FEATURES: List[str]

    # View options
    DISPLAY_ANALYSIS_SECTOIN: bool = False
    DISPLAY_SUMMARY_SECTOIN: bool = False
    DISPLAY_DETAILED_SECTION: bool = True
    DISPLAY_COMPARISON_PLOTS: bool = False
    SHOW_FIRST_COLUMN: bool = True
    TRANSPOSE_OVERALL_TABLE: bool = False
    TRANSPOSE_COMPARISON_PLOTS: bool = False
    SUMMARY_NUM_COLS: int = 3
    SUMMARY_SUBSET_INDEX: int = -1
    ITEMS_PER_PAGE: int = 5
    NUM_PROJECTS_TO_COMPARE: int = 3

    ALL_FEATURES: List[Tuple[str, List[Tuple[str, str, MetricType, str]]]] = [
        (
            "Attributes",
            [
                ("Audio", "audio", MetricType.ATTRIBUTE, ""),
                ("Text", "text", MetricType.ATTRIBUTE, ""),
                ("Text (normalized)", "text_norm", MetricType.ATTRIBUTE, ""),
                ("Transcript", "transcript", MetricType.ATTRIBUTE, ""),
                ("Text phonemes", "text_phonemes", MetricType.ATTRIBUTE, ""),
                (
                    "Transcript phonemes",
                    "transcript_phonemes",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                (
                    "Grapheme pronunciation speed",
                    "pronunciation_speed",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                (
                    "Phoneme pronunciation speed",
                    "pronunciation_speed_phonemes",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                ("Audio duration", "audio_duration", MetricType.ATTRIBUTE, ""),
                ("Speech duration", "speech_duration", MetricType.ATTRIBUTE, ""),
                ("Silence in the begining", "silence_begin", MetricType.ATTRIBUTE, ""),
                ("Silence in the end", "silence_end", MetricType.ATTRIBUTE, ""),
                (
                    "Pitch mean",
                    "pitch_mean",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                (
                    "Pitch std",
                    "pitch_std",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                (
                    "Pitch plot",
                    "pitch_plot",
                    MetricType.ATTRIBUTE,
                    "",
                ),
                ("Wavelet prosody plot", "wavelet_prosody", MetricType.ATTRIBUTE, ""),
            ],
        ),
        (
            "MOS Prediction",
            [
                ("UTMOS", "utmos", MetricType.METRIC, "↑"),
                (
                    "SpeechBERTScore",
                    "speech_bert_score",
                    MetricType.REFERENCE_AWARE_METRIC,
                    "↑",
                ),
            ],
        ),
        (
            "Speech Prosody",
            [
                ("VDE", "vde", MetricType.REFERENCE_AWARE_METRIC, "↓"),
                ("GPE", "gpe", MetricType.REFERENCE_AWARE_METRIC, "↓"),
                ("FFE", "ffe", MetricType.REFERENCE_AWARE_METRIC, "↓"),
                ("logF0 RMSE", "logf0_rmse", MetricType.REFERENCE_AWARE_METRIC, "↓"),
            ],
        ),
        (
            "Signal Quality",
            [
                ("Squim STOI", "squim_stoi", MetricType.METRIC, "↑"),
                ("Squim PESQ", "squim_pesq", MetricType.METRIC, "↑"),
                ("Squim SI-SDR", "squim_sisdr", MetricType.METRIC, "↑"),
            ],
        ),
        (
            "Speaker Similarity",
            [
                (
                    "Speaker Similarity (ECAPA-TDNN)",
                    "sim_ecapa",
                    MetricType.REFERENCE_AWARE_METRIC,
                    "↑",
                ),
            ],
        ),
        (
            "Speech Intelligibility",
            [
                ("Character distance", "character_distance", MetricType.METRIC, "↓"),
                ("Phoneme distance", "phoneme_distance", MetricType.METRIC, "↓"),
            ],
        ),
    ]

    FEATURE_ID_TO_LABEL: Dict[str, str] = {
        feature: f"{label} {sign}" if sign else label
        for (_, cat_features) in ALL_FEATURES
        for (label, feature, _, sign) in cat_features
    }

    FEATURE_LABEL_TO_ID: Dict[str, str] = {
        label: feature for (_, cat_features) in ALL_FEATURES for (label, feature, _, _) in cat_features
    }


_settings = None


def get_settings() -> Settings:
    """Retrieves the singleton settings instance, loading it if not already loaded."""
    global _settings
    if _settings is None:
        init_settings()
    return _settings


def init_settings(config_path: str = "attest/config/config.yaml"):
    """Initializes the configuration by loading it from the specified YAML file."""
    global _settings
    _settings = load_settings(config_path)


def load_settings(config_path: str) -> Settings:
    """Loads configuration data from a YAML file, validates it, and returns a Settings object."""
    with open(config_path, "rt") as f:
        data = yaml.safe_load(f)

    features_dict = data["default_view_preferences"]["features"]

    flat_data = {
        "LOGGING_LEVEL": data["logging"]["level"],
        "DATA_DIR": data["directories"]["data_dir"],
        "MODELS_DIR": data["directories"]["models_dir"],
        "FEATURES": [x for x in features_dict if features_dict[x]],
    }

    try:
        return Settings(**flat_data)
    except ValidationError as e:
        print("Configuration validation error:", e)
        raise SystemExit(f"Failed to load configuration: {e}")
