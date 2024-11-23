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

import math
from statistics import mean

from attest.src.model import Project, MetricResultEntry, MetricResult
from attest.src.utils.pitch_comparator import get_pitch_comparator
from attest.src.utils.speaker_embedder import SpeakerEmbedder, get_ecapa_embedder
from attest.src.utils.speech_bert_score import bert_score
from attest.src.utils.wavlm_large import get_wavlm_large


def get_reference_aware_metric_id_to_method():
    return {
        "vde": vde,
        "gpe": gpe,
        "ffe": ffe,
        "logf0_rmse": logf0_rmse,
        "sim_ecapa": sim_ecapa,
        "speech_bert_score": speech_bert_score,
    }


def vde(hyp_project: Project, ref_project: Project) -> MetricResult:
    pitch_comparator = get_pitch_comparator()
    scores = pitch_comparator.compare_for_projects(hyp_project, ref_project, key="VDE")
    scores = [x if not math.isnan(x) else 0.0 for x in scores]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def gpe(hyp_project: Project, ref_project: Project) -> MetricResult:
    pitch_comparator = get_pitch_comparator()
    scores = pitch_comparator.compare_for_projects(hyp_project, ref_project, key="GPE")
    scores = [x if not math.isnan(x) else 0.0 for x in scores]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def ffe(hyp_project: Project, ref_project: Project) -> MetricResult:
    pitch_comparator = get_pitch_comparator()
    scores = pitch_comparator.compare_for_projects(hyp_project, ref_project, key="FFE")
    scores = [x if not math.isnan(x) else 0.0 for x in scores]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def logf0_rmse(hyp_project: Project, ref_project: Project) -> MetricResult:
    pitch_comparator = get_pitch_comparator()
    scores = pitch_comparator.compare_for_projects(hyp_project, ref_project, key="log_F0_RMSE")
    scores = [x if not math.isnan(x) else 0.0 for x in scores]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def sim_ecapa(hyp_project: Project, ref_project: Project) -> MetricResult:
    speaker_embedder = get_ecapa_embedder()
    return speaker_similarity(speaker_embedder, hyp_project, ref_project)


def speaker_similarity(speaker_embedder: SpeakerEmbedder, hyp_project: Project, ref_project: Project) -> MetricResult:
    hyp_embeddings = speaker_embedder.extract_embeddings_for_project(hyp_project)
    ref_embeddings = speaker_embedder.extract_embeddings_for_project(ref_project)

    scores = [speaker_embedder.compute_similarity(x, y).item() for x, y in zip(hyp_embeddings, ref_embeddings)]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)


def speech_bert_score(hyp_project: Project, ref_project: Project) -> MetricResult:
    wavlm_large = get_wavlm_large()

    hyp_features = wavlm_large.extract_features_for_project(hyp_project, 14)  # layer 14
    ref_features = wavlm_large.extract_features_for_project(ref_project, 14)  # layer 14

    scores = [bert_score(x, y)[0] for x, y in zip(hyp_features, ref_features)]

    overall = mean(scores)
    detailed = [MetricResultEntry(uid, score) for uid, score in zip(hyp_project.uids, scores)]

    return MetricResult(overall=overall, detailed=detailed)
