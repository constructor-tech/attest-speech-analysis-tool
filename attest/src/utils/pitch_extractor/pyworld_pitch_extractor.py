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
from attest.src.utils.performance_tracker import PerformanceTracker
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
        self.hop_length_seconds = 1.0 / self.fps
        self.fmin = 50
        self.fmax = 550

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/pyworld/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project(self, project: Project):
        tracker = PerformanceTracker(name="Computing pitch values using pyworld", start=True)
        pitch_values = [self._compute_pitch_values(x) for x in project.audio_files]
        tracker.end()

        return pitch_values

    def _compute_pitch_values(self, audio_file: str):
        # TODO @od 25.11.2024: add outlier removal
        wav, sr = librosa.load(audio_file, sr=None)
        pitch = self._compute_pitch(wav, sr)
        pitch[pitch == 0] = np.nan
        return pitch

    def _compute_pitch(self, wav, sr):
        wav = wav.astype(np.float64)
        f0, time_axis = pw.harvest(
            wav,
            sr,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=self.hop_length_seconds * 1000,
        )
        return f0
