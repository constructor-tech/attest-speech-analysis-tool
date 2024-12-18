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
import torchaudio
from functools import lru_cache


def load_audio_tensor(audio_file, target_sr=None, target_channels=1, device="cpu"):
    audio, sr = torchaudio.load(audio_file)

    # Mix to mono if necessary
    if audio.shape[0] > 1 and target_channels == 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Expand channels if necessary
    if audio.shape[0] == 1 and target_channels == 2:
        audio = audio.expand(target_channels, audio.shape[1])

    # Resample if necessary
    if target_sr is not None and sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    audio = audio.to(device)
    return audio, sr


@lru_cache(None)
def _get_audio_attributes(file_path):
    y, sr = librosa.load(file_path, sr=None)
    _, index = librosa.effects.trim(y, top_db=40)
    duration = float(y.shape[0] / sr)
    duration_speech = float((index[1] - index[0]) / sr)
    silence_begin = float(index[0] / sr)
    silence_end = float((y.shape[0] - index[1]) / sr)
    return {
        "duration": duration,
        "duration_speech": duration_speech,
        "silence_begin": silence_begin,
        "silence_end": silence_end,
    }


def get_audio_duration(file_path):
    return _get_audio_attributes(file_path)["duration"]


def get_speech_duration(file_path):
    return _get_audio_attributes(file_path)["duration_speech"]


def get_silence_begin(file_path):
    return _get_audio_attributes(file_path)["silence_begin"]


def get_silence_end(file_path):
    return _get_audio_attributes(file_path)["silence_end"]
