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

import os
import torch
from transformers import WavLMModel

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.audio_utils import load_audio_tensor
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.performance_tracker import PerformanceTracker


_wavlm_large = None


def get_wavlm_large():
    global _wavlm_large
    if _wavlm_large is None:
        _wavlm_large = WavLMLarge()

    return _wavlm_large


settings = get_settings()


class WavLMLarge:

    def __init__(self):
        self.model = None
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "wavlm", settings.WAVLM_MODEL_NAME)
        self.sampling_rate = 16000
        self.device = settings.DEVICE

    def load_model(self):
        if self.model is None:
            tracker = PerformanceTracker(name="Loading wavlm-large model", start=True)
            self.model = WavLMModel.from_pretrained(
                settings.WAVLM_MODEL_NAME,
                cache_dir=self.model_cache_dir,
            ).to(self.device)
            tracker.end()

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/wavlm_large_features_layer_${{2}}/features.pt",
        method="torch",
        validator=validate_matching_to_project_size,
    )
    def extract_features_for_project(self, project: Project, layer: int = -1):
        self.load_model()

        tracker = PerformanceTracker(name="Extracting WavLM features", start=True)
        features = []
        with torch.no_grad():
            for audio_path in project.audio_files:
                audio_tensor, _ = load_audio_tensor(audio_path, target_sr=self.sampling_rate, target_channels=1, device=self.device)
                feats = self.model(audio_tensor, output_hidden_states=True).hidden_states[layer]
                features.append(feats.squeeze(0).cpu())
        tracker.end()

        return features
