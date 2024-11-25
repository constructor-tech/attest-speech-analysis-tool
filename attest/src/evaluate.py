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

from attest.src.features import compute_reference_free_feature, is_feature
from attest.src.model import load_project, EvaluationResult
from attest.src.settings import get_settings
from typing import List, Dict


def evaluate(project: str, features: List[str], feature_params: Dict[str, str]) -> EvaluationResult:
    settings = get_settings()
    settings.apply_feature_params(feature_params)
    cache_filenames = settings.get_feature_basenames(features)

    project = load_project(project)
    output = EvaluationResult(project=project.name, features=features)

    for feature_id, cache_filename in zip(features, cache_filenames):
        if is_feature(feature_id):
            feature_result = compute_reference_free_feature(feature_id, project, cache_filename)
            if feature_result:
                output.results[feature_id] = feature_result

    return output
