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
import parselmouth

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.performance_tracker import PerformanceTracker
from .pitch_extractor import PitchExtractor


_parselmouth_pitch_extractor = None


def get_parselmouth_pitch_extractor():
    global _parselmouth_pitch_extractor
    if _parselmouth_pitch_extractor is None:
        _parselmouth_pitch_extractor = ParselmouthPitchExtractor()

    return _parselmouth_pitch_extractor


settings = get_settings()


class ParselmouthPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch/parselmouth/values.pkl",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def compute_pitch_values_for_project(self, project: Project):
        tracker = PerformanceTracker(name="Computing pitch values using praat-parselmouth", start=True)
        pitch_values = [self._compute_pitch_values(x) for x in project.audio_files]
        tracker.end()

        return pitch_values

    def _compute_pitch_values(self, audio_file: str):
        wav, sr = librosa.load(audio_file, sr=None)
        n_points = len(wav) // (sr / self.fps)
        pitch = self._compute_pitch(wav, sr, n_points)
        pitch = self._remove_outliers(pitch)
        pitch[pitch == 0] = np.nan
        return pitch

    def _compute_pitch(self, wav, sr, size):
        snd = parselmouth.Sound(wav, sampling_frequency=sr)
        delta = 3
        time_step = snd.duration / (size + delta)
        pitch = snd.to_pitch(time_step=time_step).selected_array["frequency"]
        while pitch.shape[0] < size:
            delta += 1
            time_step = snd.duration / (size + delta)
            pitch = snd.to_pitch(time_step=time_step).selected_array["frequency"]
        while pitch.shape[0] > size:
            delta -= 1
            time_step = snd.duration / (size + delta)
            pitch = snd.to_pitch(time_step=time_step).selected_array["frequency"]
        return pitch

    def _remove_outliers(self, pitch_array):
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
