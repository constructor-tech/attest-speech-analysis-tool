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
from statistics import mean
from torchmetrics.text import CharErrorRate, WordErrorRate

from attest.src.model import Project, MetricResultEntry, MetricResult
from attest.src.settings import get_settings
from attest.src.utils.whisper import get_whisper
from attest.src.utils.edit_distance import (
    edit_distance_many,
    format_text_for_edit_distance,
)
from attest.src.utils.librosa_utils import (
    get_audio_duration,
    get_speech_duration,
    get_silence_begin,
    get_silence_end,
)
from attest.src.utils.phonemizer import get_phonemizer
from attest.src.utils.pitch_extractor import get_pitch_extractor
from attest.src.utils.text_normalizer import get_text_normalizer
from attest.src.utils.squim import get_squim
from attest.src.utils.utmos import get_utmos_strong


def get_metric_id_to_method():
    return {
        "cer": cer,
        "wer": wer,
        "per": per,
        "character_distance": character_distance,
        "phoneme_distance": phoneme_distance,
        "pronunciation_speed": pronunciation_speed,
        "pronunciation_speed_phonemes": pronunciation_speed_phonemes,
        "audio_duration": audio_duration,
        "speech_duration": speech_duration,
        "silence_begin": silence_begin,
        "silence_end": silence_end,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "utmos": utmos,
        "squim_stoi": squim_stoi,
        "squim_pesq": squim_pesq,
        "squim_sisdr": squim_sisdr,
    }


def cer(project: Project) -> MetricResult:
    asr = get_whisper()
    metric = CharErrorRate()
    transcriptions = asr.transcribe_project(project)

    text_normalizer = get_text_normalizer()
    texts_norm = text_normalizer.normalize_project(project)

    texts_for_cer = [format_text_for_edit_distance(x) for x in texts_norm]
    transcriptions_for_cer = [format_text_for_edit_distance(x) for x in transcriptions]

    result = [metric(seq1, seq2).item() for seq1, seq2 in zip(transcriptions_for_cer, texts_for_cer)]
    overall = mean(result)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, result)]
    return MetricResult(overall=overall, detailed=detailed)


def wer(project: Project) -> MetricResult:
    asr = get_whisper()
    metric = WordErrorRate()
    transcriptions = asr.transcribe_project(project)

    text_normalizer = get_text_normalizer()
    texts_norm = text_normalizer.normalize_project(project)

    texts_for_cer = [format_text_for_edit_distance(x, remove_spaces=False) for x in texts_norm]
    transcriptions_for_cer = [format_text_for_edit_distance(x, remove_spaces=False) for x in transcriptions]

    result = [metric(seq1, seq2).item() for seq1, seq2 in zip(transcriptions_for_cer, texts_for_cer)]
    overall = mean(result)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, result)]
    return MetricResult(overall=overall, detailed=detailed)


def per(project: Project) -> MetricResult:
    settings = get_settings()
    asr = get_whisper()
    metric = CharErrorRate()
    text_normalizer = get_text_normalizer()
    phonemizer = get_phonemizer()

    transcriptions = asr.transcribe_project(project)
    transcriptions_phonemes = phonemizer.phonemize_many(
        transcriptions, f"{project.name}/g2p/phonemes-{settings.WHISPER_MODEL_NAME}-{settings.PHONEMIZATION_METHOD}.txt"
    )

    texts_norm = text_normalizer.normalize_project(project)
    texts_phonemes = phonemizer.phonemize_many(
        texts_norm, f"{project.name}/g2p/phonemes-{settings.TEXT_NORM_METHOD}-{settings.PHONEMIZATION_METHOD}.txt"
    )

    texts_for_cer = [format_text_for_edit_distance(x) for x in texts_phonemes]
    transcriptions_for_cer = [format_text_for_edit_distance(x) for x in transcriptions_phonemes]

    result = [metric(seq1, seq2).item() for seq1, seq2 in zip(transcriptions_for_cer, texts_for_cer)]
    overall = mean(result)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, result)]
    return MetricResult(overall=overall, detailed=detailed)


def character_distance(project: Project) -> MetricResult:
    asr = get_whisper()
    transcriptions = asr.transcribe_project(project)

    text_normalizer = get_text_normalizer()
    texts_norm = text_normalizer.normalize_project(project)

    texts_for_cer = [format_text_for_edit_distance(x) for x in texts_norm]
    transcriptions_for_cer = [format_text_for_edit_distance(x) for x in transcriptions]

    result = edit_distance_many(texts_for_cer, transcriptions_for_cer)
    overall = mean(result)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, result)]
    return MetricResult(overall=overall, detailed=detailed)


def phoneme_distance(project: Project) -> MetricResult:
    settings = get_settings()
    asr = get_whisper()
    text_normalizer = get_text_normalizer()
    phonemizer = get_phonemizer()

    texts_norm = text_normalizer.normalize_project(project)
    texts_phonemes = phonemizer.phonemize_many(
        texts_norm, f"{project.name}/g2p/phonemes-{settings.TEXT_NORM_METHOD}-{settings.PHONEMIZATION_METHOD}.txt"
    )

    transcriptions = asr.transcribe_project(project)
    transcriptions_phonemes = phonemizer.phonemize_many(
        transcriptions, f"{project.name}/g2p/phonemes-{settings.WHISPER_MODEL_NAME}-{settings.PHONEMIZATION_METHOD}.txt"
    )

    texts_for_cer = [format_text_for_edit_distance(x) for x in texts_phonemes]
    transcriptions_for_cer = [format_text_for_edit_distance(x) for x in transcriptions_phonemes]

    result = edit_distance_many(texts_for_cer, transcriptions_for_cer)
    overall = mean(result)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, result)]
    return MetricResult(overall=overall, detailed=detailed)


def pitch_mean(project: Project) -> MetricResult:
    pitch_extractor = get_pitch_extractor()
    pitch_values = pitch_extractor.compute_pitch_values_for_project(project)

    overall = np.nanmean(np.concatenate(pitch_values))
    detailed = []
    for uid, values in zip(project.uids, pitch_values):
        p_mean = 0 if np.isnan(values).all() else np.nanmean(values)
        detailed.append(MetricResultEntry(uid, p_mean))
    return MetricResult(overall=overall, detailed=detailed)


def pitch_std(project: Project) -> MetricResult:
    pitch_extractor = get_pitch_extractor()
    pitch_values = pitch_extractor.compute_pitch_values_for_project(project)

    overall = np.nanstd(np.concatenate(pitch_values))
    detailed = []
    for uid, values in zip(project.uids, pitch_values):
        p_std = 0 if np.isnan(values).all() else np.nanstd(values)
        detailed.append(MetricResultEntry(uid, p_std))
    return MetricResult(overall=overall, detailed=detailed)


def audio_duration(project: Project) -> MetricResult:
    total = 0
    detailed = []

    for uid, audio_file in zip(project.uids, project.audio_files):
        duration = get_audio_duration(audio_file)
        total += duration
        detailed.append(MetricResultEntry(uid, duration))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def speech_duration(project: Project) -> MetricResult:
    total = 0
    detailed = []

    for uid, audio_file in zip(project.uids, project.audio_files):
        duration = get_speech_duration(audio_file)
        total += duration
        detailed.append(MetricResultEntry(uid, duration))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def silence_begin(project: Project) -> MetricResult:
    total = 0
    detailed = []

    for uid, audio_file in zip(project.uids, project.audio_files):
        silence = get_silence_begin(audio_file)
        total += silence
        detailed.append(MetricResultEntry(uid, silence))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def silence_end(project: Project) -> MetricResult:
    total = 0
    detailed = []

    for uid, audio_file in zip(project.uids, project.audio_files):
        silence = get_silence_end(audio_file)
        total += silence
        detailed.append(MetricResultEntry(uid, silence))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def pronunciation_speed(project: Project) -> MetricResult:
    total = 0
    detailed = []

    for uid, audio_file, text in zip(project.uids, project.audio_files, project.texts):
        duration_in_seconds = get_speech_duration(audio_file)
        pron_speed = len(text) / duration_in_seconds
        total += pron_speed
        detailed.append(MetricResultEntry(uid, pron_speed))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def pronunciation_speed_phonemes(project: Project) -> MetricResult:
    settings = get_settings()
    text_normalizer = get_text_normalizer()
    phonemizer = get_phonemizer()
    texts_norm = text_normalizer.normalize_project(project)
    texts_norm_phonemes = phonemizer.phonemize_many(
        texts_norm, f"{project.name}/g2p/phonemes-{settings.TEXT_NORM_METHOD}-{settings.PHONEMIZATION_METHOD}.txt"
    )

    total = 0
    detailed = []

    for uid, audio_file, text in zip(project.uids, project.audio_files, texts_norm_phonemes):
        duration_in_seconds = get_speech_duration(audio_file)
        pron_speed = len(text) / duration_in_seconds
        total += pron_speed
        detailed.append(MetricResultEntry(uid, pron_speed))

    overall = total / len(project.audio_files)
    return MetricResult(overall=overall, detailed=detailed)


def utmos(project: Project) -> MetricResult:
    predictor = get_utmos_strong()
    scores = predictor.predict_project(project)

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def squim_stoi(project: Project) -> MetricResult:
    predictor = get_squim()
    scores = predictor.predict_project(project, key="STOI")

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def squim_pesq(project: Project) -> MetricResult:
    predictor = get_squim()
    scores = predictor.predict_project(project, key="PESQ")

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def squim_sisdr(project: Project) -> MetricResult:
    predictor = get_squim()
    scores = predictor.predict_project(project, key="SI-SDR")

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)
