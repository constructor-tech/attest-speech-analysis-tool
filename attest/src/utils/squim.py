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

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from attest.src.utils.performance_tracker import PerformanceTracker


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
        self.objective_model = None
        self.subjective_model = None
        self.device = settings.DEVICE


    def load_objective_model(self):
        if self.objective_model is None:
            tracker = PerformanceTracker(name="Loading squim objective model", start=True)
            self.objective_model = SQUIM_OBJECTIVE.get_model()
            self.objective_model.to(self.device)
            tracker.end()


    def load_subjective_model(self):
        if self.subjective_model is None:
            tracker = PerformanceTracker(name="Loading squim subjective model", start=True)
            self.subjective_model = SQUIM_SUBJECTIVE.get_model()
            self.subjective_model.to(self.device)
            tracker.end()


    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/squim/squim_subjective_scores.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def predict_project_subjective(self, hyp_project: Project, ref_project: Project):
        scores = []
        self.load_subjective_model()
        self.logger.info("Computing scores...")
        for audio_hyp_path, audio_ref_path in zip(hyp_project.audio_files, ref_project.audio_files):
            scores.append(self.predict_subjective(audio_hyp_path, audio_ref_path))
        self.logger.info("Computing scores is done!")
        return [x for x in scores]


    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/squim/squim_objective_scores.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def predict_project_objective(self, project: Project, key: str):
        scores = []
        self.load_objective_model()

        tracker = PerformanceTracker(name="Computing squim objective scores", start=True)
        for audio_path in project.audio_files:
            scores.append(self.predict_objective(audio_path))
        tracker.end()
        return [x[key] for x in scores]


    def predict_subjective(self, audio_hyp_path, audio_ref_path):
        wave_hyp, _ = self._load_audio(audio_hyp_path, 16000)
        wave_ref, _ = self._load_audio(audio_ref_path, 16000)
        with torch.no_grad():
            mos = self.subjective_model(wave_hyp.to(self.device), wave_ref.to(self.device))
        return mos.cpu().item()


    def predict_objective(self, audio_path):
        wave, _ = self._load_audio(audio_path, 16000)
        with torch.no_grad():
            stoi, pesq, si_sdr = self.objective_model(wave.to(self.device))
        return {
            "STOI": stoi.cpu().item(),
            "PESQ": pesq.cpu().item(),
            "SI-SDR": si_sdr.cpu().item(),
        }
    

    def _load_audio(self, audio_path, target_sr):
        wave, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            wave = F.resample(wave, sr, target_sr)
        return wave, target_sr

