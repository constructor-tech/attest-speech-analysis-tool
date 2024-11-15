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

import time
from typing import Union

from attest.src.settings import get_settings
from attest.src.model import Project, AttributeResult, MetricResult
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import (
    validate_feature_from_cache,
    validate_pairwise_feature_from_cache,
)
from attest.src.utils.logger import get_logger
from .attributes import (
    audio,
    text,
    text_norm,
    text_phonemes,
    transcript,
    transcript_phonemes,
    pitch_plot,
    wavelet_prosody,
)

from .metrics import (
    cer,
    wer,
    per,
    character_distance,
    phoneme_distance,
    pronunciation_speed,
    pronunciation_speed_phonemes,
    audio_duration,
    speech_duration,
    silence_begin,
    silence_end,
    pitch_mean,
    pitch_std,
    utmos,
    squim_stoi,
    squim_pesq,
    squim_sisdr,
)

from .pairwise_metrics import (
    vde,
    gpe,
    ffe,
    logf0_rmse,
    sim_ecapa,
    speech_bert_score,
)


logger = get_logger()
settings = get_settings()


_attribute_id_to_method = {
    "audio": audio,
    "text": text,
    "text_norm": text_norm,
    "text_phonemes": text_phonemes,
    "transcript": transcript,
    "transcript_phonemes": transcript_phonemes,
    "pitch_plot": pitch_plot,
    "wavelet_prosody": wavelet_prosody,
}

_metric_id_to_method = {
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

_pairwise_metric_id_to_method = {
    "vde": vde,
    "gpe": gpe,
    "ffe": ffe,
    "logf0_rmse": logf0_rmse,
    "sim_ecapa": sim_ecapa,
    "speech_bert_score": speech_bert_score,
}

_feature_id_to_method = {
    **_attribute_id_to_method,
    **_metric_id_to_method,
}

_pairwise_feature_id_to_method = {
    **_pairwise_metric_id_to_method,
}


def is_attribute(feature_id: str) -> bool:
    return feature_id in _attribute_id_to_method


def is_metric(feature_id: str) -> bool:
    # TODO: bad name, is_pairwise_metric is also metrics
    return feature_id in _metric_id_to_method


def is_pairwise_metric(feature_id: str) -> bool:
    return feature_id in _pairwise_metric_id_to_method


def is_feature(feature_id: str) -> bool:
    # TODO: bad name, pairwise_feature is also feature
    return feature_id in _feature_id_to_method


def is_pairwise_feature(feature_id: str) -> bool:
    return feature_id in _pairwise_feature_id_to_method


@CacheHandler(
    cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/features1/${{2}}.pickle",
    method="pickle",
    validator=validate_feature_from_cache,
)
def compute_feature(feature_id: str, project: Project, cache_filename: str) -> Union[AttributeResult, MetricResult]:
    logger.info('Computing feature "%s" for project "%s"...' % (feature_id, project.name))

    start_time = time.time()
    feature = _feature_id_to_method.get(feature_id)
    if feature is None:
        logger.info('Unknown feature "%s", skipping.' % feature_id)
        return None

    feature_result = feature(project)
    end_time = time.time()
    logger.info('Computation time for feature "%s": %.2f seconds.' % (feature_id, end_time - start_time))

    return feature_result


@CacheHandler(
    cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/features2/${{2.name}}/${{3}}.pickle",
    method="pickle",
    validator=validate_pairwise_feature_from_cache,
)
def compute_pairwise_feature(
    feature_id: str, hyp_project: Project, ref_project: Project, cache_filename: str
) -> Union[AttributeResult, MetricResult]:
    logger.info(
        'Computing feature "%s" for projects "%s" (hyp), "%s" (ref)...'
        % (feature_id, hyp_project.name, ref_project.name)
    )

    start_time = time.time()
    feature = _pairwise_feature_id_to_method.get(feature_id)
    if feature is None:
        logger.info('Unknown feature "%s", skipping.' % feature_id)
        return None

    feature_result = feature(hyp_project, ref_project)
    end_time = time.time()
    logger.info('Computation time for feature "%s": %.2f seconds.' % (feature_id, end_time - start_time))

    return feature_result
