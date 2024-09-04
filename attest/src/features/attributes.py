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

from attest.src.model import Project, AttributeResultEntry, AttributeResult
from attest.src.settings import get_settings
from attest.src.utils.whisper import get_whisper
from attest.src.utils.phonemizer import get_phonemizer
from attest.src.utils.pitch_extractor import get_pitch_extractor
from attest.src.utils.text_normalizer import get_text_normalizer
from attest.src.utils.wavelet_prosody import get_wavelet_prosody_extractor


def audio(project: Project) -> AttributeResult:
    detailed = [
        AttributeResultEntry(uid, audio_path=audio_path, message=uid)
        for uid, audio_path in zip(project.uids, project.audio_files)
    ]
    return AttributeResult(detailed=detailed)


def text(project: Project) -> AttributeResult:
    detailed = [AttributeResultEntry(uid, message=text) for uid, text in zip(project.uids, project.texts)]
    return AttributeResult(detailed=detailed)


def text_norm(project: Project) -> AttributeResult:
    text_normalizer = get_text_normalizer()
    texts_norm = text_normalizer.normalize_project(project)
    detailed = [AttributeResultEntry(uid, message=text) for uid, text in zip(project.uids, texts_norm)]
    return AttributeResult(detailed=detailed)


def transcript(project: Project) -> AttributeResult:
    asr = get_whisper()
    transcriptions = asr.transcribe_project(project)
    detailed = [AttributeResultEntry(uid, message=text) for uid, text in zip(project.uids, transcriptions)]
    return AttributeResult(detailed=detailed)


def text_phonemes(project: Project) -> AttributeResult:
    text_normalizer = get_text_normalizer()
    texts_norm = text_normalizer.normalize_project(project)

    settings = get_settings()
    cache_path = f"{project.name}/g2p/phonemes-{settings.TEXT_NORM_METHOD}.txt"

    phonemizer = get_phonemizer()
    phonemes = phonemizer.phonemize_many(texts_norm, cache_path)

    detailed = [AttributeResultEntry(uid, message=text) for uid, text in zip(project.uids, phonemes)]
    return AttributeResult(detailed=detailed)


def transcript_phonemes(project: Project) -> AttributeResult:
    asr = get_whisper()
    transcriptions = asr.transcribe_project(project)

    settings = get_settings()
    cache_path = f"{project.name}/g2p/phonemes-{settings.WHISPER_MODEL_NAME}.txt"

    phonemizer = get_phonemizer()
    phonemes = phonemizer.phonemize_many(transcriptions, cache_path)
    detailed = [AttributeResultEntry(uid, message=text) for uid, text in zip(project.uids, phonemes)]
    return AttributeResult(detailed=detailed)


def pitch_plot(project: Project) -> AttributeResult:
    settings = get_settings()
    pitch_extractor = get_pitch_extractor()
    pitch_values = pitch_extractor.compute_pitch_values_for_project(project)

    # for word-level pitch
    forced_aliger = get_whisper()
    word_alignments = forced_aliger.align_project(project)

    detailed = []
    for uid, values, ali in zip(project.uids, pitch_values, word_alignments):
        # pitch values for plot
        plot_values = values.tolist()
        plot_values = ["NaN" if np.isnan(v) else v for v in plot_values]

        min_val = 0 if np.isnan(values).all() else np.nanmin(values)

        # word boundaries for plot
        plot_labels = []  # text, x, y
        horizontal_segments = []  # y x0 x1

        for word, start, end in ali:
            scale = pitch_extractor.fps
            scaled_start = int(scale * start)
            scaled_end = int(scale * end)
            y_val = np.nanmean(values[scaled_start:scaled_end])
            if np.isnan(y_val):
                y_val = min_val
            plot_labels.append((word, (scaled_start + scaled_end) / 2, y_val + 3))
            horizontal_segments.append((float(y_val), scaled_start, scaled_end))

        plot = {
            "title": f"Pitch values (computed with {settings.PITCH_EXTRACT_METHOD})",
            "values": plot_values,
            "plot_labels": plot_labels,
            "horizontal_segments": horizontal_segments,
        }

        detailed.append(AttributeResultEntry(uid, plot_data=plot))

    return AttributeResult(detailed=detailed)


def wavelet_prosody(project: Project) -> AttributeResult:
    forced_aliger = get_whisper()
    wavelet_prosody_extractor = get_wavelet_prosody_extractor()

    word_alignments = forced_aliger.align_project(project)

    output = wavelet_prosody_extractor.extract_features_for_project(project, word_alignments)

    detailed = []
    for uid, entry in zip(project.uids, output):
        detailed.append(AttributeResultEntry(uid, image_path=entry["image"]))
    return AttributeResult(detailed=detailed)
