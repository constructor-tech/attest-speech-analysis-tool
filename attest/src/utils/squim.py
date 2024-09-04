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
import pickle
import time
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.logger import get_logger


_predictor = None


def get_squim():
    global _predictor
    if _predictor is None:
        _predictor = SquimPredictor()

    return _predictor


settings = get_settings()


class SquimPredictor:

    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.device = settings.DEVICE

    def load_predictor(self):
        if self.model is None:
            start_time = time.time()
            self.logger.info("Loading squim objective model...")

            self.model = SQUIM_OBJECTIVE.get_model()  # TODO cache
            self.model.to(self.device)
            self.logger.info("Loaded squim objective model in %.2f seconds" % (time.time() - start_time))

    def predict_project(self, project: Project, key: str):
        scores = []

        cache_file = os.path.join(settings.CACHE_DIR, project.name, "squim/squim_scores.pickle")

        if os.path.exists(cache_file):
            self.logger.info('Found cached file with scores: "%s"' % cache_file)
            with open(cache_file, "rb") as f:
                scores = pickle.load(f)

        if len(scores) == len(project.audio_files):
            self.logger.info("Used scores from cache!")

        else:
            self.load_predictor()
            self.logger.info("Computing scores...")
            for audio_path in project.audio_files:
                scores.append(self.predict(audio_path))
            self.logger.info("Computing scores is done!")

        return [x[key] for x in scores]

    def predict(self, audio_path):
        wave, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wave = F.resample(wave, sr, 16000)
        with torch.no_grad():
            stoi, pesq, si_sdr = self.model(wave.to(self.device))
        return {
            "STOI": stoi.cpu().item(),
            "PESQ": pesq.cpu().item(),
            "SI-SDR": si_sdr.cpu().item(),
        }
