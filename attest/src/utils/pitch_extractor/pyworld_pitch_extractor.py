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
import numpy as np
import pyworld as pw

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from .pitch_extractor import PitchExtractor


_pyworld_pitch_extractor = None


def get_pyworld_pitch_extractor():
    global _pyworld_pitch_extractor
    if _pyworld_pitch_extractor is None:
        _pyworld_pitch_extractor = PyworldPitchExtractor()

    return _pyworld_pitch_extractor


settings = get_settings()


class PyworldPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self.hop_length_seconds = 1.0 / self.fps
        self.fmin = 50
        self.fmax = 550

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/pyworld/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project(self, project: Project):
        pitch_values = []

        self.logger.info("Computing pitch values...")
        for x in project.audio_files:
            pitch_values.append(self._compute_pitch_values(x))
        self.logger.info("Computing pitch values is done!")

        return pitch_values

    def _compute_pitch_values(self, audio_file: str):
        wav, sr = librosa.load(audio_file, sr=None)
        pitch = self._compute_pitch(wav, sr, len(wav) // (sr / self.fps))
        print(audio_file, pitch.shape)
        # pitch = self._remove_outliers(pitch)
        pitch[pitch == 0] = np.nan
        return pitch

    def _compute_pitch(self, wav, sr, size):
        wav = wav.astype(np.float64)
        f0, time_axis = pw.harvest(
            wav,
            sr,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=self.hop_length_seconds * 1000,
        )
        return f0

    def _remove_outliers(self, pitch_array):
        # TODO @od 25.10.2023: Code duplicated in pitch_comparator.py
        pitch_mean = np.mean(pitch_array[pitch_array != 0])
        i = 0
        n = len(pitch_array)
        while i < n:
            if pitch_array[i] == 0:
                i += 1
                continue

            i0, i1 = i, i + 1
            local_pitch_sum = pitch_array[i]
            local_pitch_mean = pitch_array[i]
            while i1 < n and pitch_array[i1] != 0 and 0.66 < pitch_array[i1] / local_pitch_mean < 1.5:
                local_pitch_sum += pitch_array[i1]
                local_pitch_mean = local_pitch_sum / (i1 + 1 - i0)
                i1 += 1

            if not 0.66 < local_pitch_mean / pitch_mean < 1.5:
                pitch_array[i0:i1] = 0

            i = i1
        return pitch_array
