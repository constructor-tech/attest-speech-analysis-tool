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
import torch
from pydantic import BaseModel, ValidationError
from typing import List, Dict


class Settings(BaseModel):
    LOGGING_LEVEL: str
    DATA_DIR: str
    CACHE_DIR: str
    MODELS_DIR: str
    THIRD_PARTY_DIR: str
    DEVICE: str
    PITCH_EXTRACT_METHOD: str
    TEXT_NORM_METHOD: str
    WAVLM_MODEL_NAME: str
    WHISPER_MODEL_NAME: str
    WHISPER_USE_FP16: bool

    def apply_feature_params(self, feature_params: Dict[str, str]):
        if "pitch_extraction_method" in feature_params:
            self.PITCH_EXTRACT_METHOD = feature_params["pitch_extraction_method"]
        if "text_normalization_method" in feature_params:
            self.TEXT_NORM_METHOD = feature_params["text_normalization_method"]

    def get_feature_basenames(self, features: List[str]):
        output = []
        for feature_id in features:
            if feature_id in ["text_norm"]:
                output.append(f"{feature_id}-{self.TEXT_NORM_METHOD}")
            elif feature_id in ["cer", "wer", "per", "character_distance", "phoneme_distance"]:
                output.append(f"{feature_id}-{self.WHISPER_MODEL_NAME}-{self.TEXT_NORM_METHOD}")
            elif feature_id in ["pitch_mean", "pitch_std", "pitch_plot"]:
                output.append(f"{feature_id}-{self.PITCH_EXTRACT_METHOD}")
            elif feature_id in ["vde", "gpe", "ffe", "logf0_rmse"]:
                output.append(f"{feature_id}-{self.PITCH_EXTRACT_METHOD}")
            else:
                output.append(feature_id)
        return output


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

    flat_data = {
        "LOGGING_LEVEL": data["logging"]["level"],
        "DATA_DIR": data["directories"]["data_dir"],
        "CACHE_DIR": data["directories"]["cache_dir"],
        "MODELS_DIR": data["directories"]["models_dir"],
        "THIRD_PARTY_DIR": data["directories"]["third_party_dir"],
        "DEVICE": get_device(data["device"]),
        "PITCH_EXTRACT_METHOD": data["feature_params"]["pitch_extraction_method"],
        "TEXT_NORM_METHOD": data["feature_params"]["text_normalization_method"],
        "WAVLM_MODEL_NAME": data["models"]["wavlm"]["model_name"],
        "WHISPER_MODEL_NAME": data["models"]["whisper"]["model_name"],
        "WHISPER_USE_FP16": data["models"]["whisper"]["use_fp16"],
    }

    try:
        return Settings(**flat_data)
    except ValidationError as e:
        raise SystemExit(f"Failed to load configuration: {e}")


def get_device(desired_device: str) -> str:
    """Determines the device to use based on the configuration and system capabilities."""
    if desired_device.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
