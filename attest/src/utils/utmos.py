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

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.audio_utils import load_audio_tensor
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.performance_tracker import PerformanceTracker


_predictor = None


def get_utmos_strong():
    global _predictor
    if _predictor is None:
        _predictor = UTMOSPredictor()

    return _predictor


settings = get_settings()


class UTMOSPredictor:

    def __init__(self):
        self.predictor = None
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "utmos")
        self.device = settings.DEVICE
        self.sampling_rate = 16000

    def load_predictor(self):
        if self.predictor is None:
            tracker = PerformanceTracker(name="Loading utmos22_strong model", start=True)
            torch.hub.set_dir(self.model_cache_dir)
            self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.predictor.to(self.device)
            tracker.end()

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/utmos/utmos_scores.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def predict_project(self, project: Project):
        self.load_predictor()

        tracker = PerformanceTracker(name="Computing utmos scores", start=True)
        scores = []
        for audio_path in project.audio_files:
            scores.append(self.predict(audio_path))
        tracker.end()

        return scores

    def predict(self, audio_path):
        audio_tensor, _ = load_audio_tensor(
            audio_path, target_sr=self.sampling_rate, target_channels=1, device=self.device
        )
        with torch.no_grad():
            score = self.predictor(audio_tensor, self.sampling_rate)
        return score.item()
