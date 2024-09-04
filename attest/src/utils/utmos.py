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

import librosa
import os
import time
import torch

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger


_predictor = None


def get_utmos_strong():
    global _predictor
    if _predictor is None:
        _predictor = UTMOSPredictor()

    return _predictor


settings = get_settings()


class UTMOSPredictor:

    def __init__(self):
        self.logger = get_logger()
        self.predictor = None
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "utmos")
        self.device = settings.DEVICE

    def load_predictor(self):
        if self.predictor is None:
            start_time = time.time()
            self.logger.info("Loading utmos22_strong model...")

            torch.hub.set_dir(self.model_cache_dir)
            self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.predictor.to(self.device)
            self.logger.info("Loaded utmos22_strong model in %.2f seconds" % (time.time() - start_time))

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/utmos/utmos_scores.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def predict_project(self, project: Project):
        scores = []
        self.load_predictor()
        self.logger.info("Computing scores...")
        for x in project.audio_files:
            scores.append(self.predict(x))
        self.logger.info("Computing scores is done!")
        return scores

    def predict(self, audio_path):
        wave, sr = librosa.load(audio_path, sr=None)
        with torch.no_grad():
            x = torch.from_numpy(wave).unsqueeze(0).to(self.device)
            score = self.predictor(x, sr)
        return score.item()
