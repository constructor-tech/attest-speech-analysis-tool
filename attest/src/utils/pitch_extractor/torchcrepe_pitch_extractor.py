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

import numpy as np
import torchcrepe

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from .pitch_extractor import PitchExtractor


_torchcrepe_pitch_extractor = None


def get_torchcrepe_pitch_extractor(model_name):
    global _torchcrepe_pitch_extractor
    if _torchcrepe_pitch_extractor is None:
        _torchcrepe_pitch_extractor = CrepePitchExtractor(model_name)

    return _torchcrepe_pitch_extractor


settings = get_settings()


class CrepePitchExtractor(PitchExtractor):

    def __init__(self, model_name):
        super().__init__()
        self.logger = get_logger()
        self.hop_length_seconds = 1.0 / self.fps
        self.fmin = 50
        self.fmax = 550
        self.model = model_name
        self.device = settings.DEVICE
        self.batch_size = 2048

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/torchcrepe/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project(self, project: Project):
        self.logger.info("Computing pitch values...")

        if self.model == "full":
            pitch_values = self.compute_pitch_values_for_project_model_full(project)
        else:
            pitch_values = self.compute_pitch_values_for_project_model_tiny(project)

        self.logger.info("Computing pitch values is done!")
        return pitch_values

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/torchcrepe-tiny/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project_model_tiny(self, project: Project):
        return [self._compute_pitch_values(x) for x in project.audio_files]

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/torchcrepe-full/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project_model_full(self, project: Project):
        return [self._compute_pitch_values(x) for x in project.audio_files]

    def _compute_pitch_values(self, audio_file: str):
        audio, sr = torchcrepe.load.audio(audio_file)

        hop_length = int(sr * self.hop_length_seconds)

        pitch = torchcrepe.predict(
            audio,
            sr,
            hop_length,
            self.fmin,
            self.fmax,
            self.model,
            batch_size=self.batch_size,
            device=self.device,
        )
        return pitch.squeeze(0).cpu().numpy().astype(np.float64)
