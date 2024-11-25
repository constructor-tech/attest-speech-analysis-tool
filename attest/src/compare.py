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

from attest.src.features import (
    is_reference_aware_feature,
    compute_reference_aware_feature,
    compute_reference_free_feature,
)
from attest.src.model import (
    load_project,
    validate_project_alignment,
    ComparisonResult,
    FeatureComparisonResult,
)
from attest.src.settings import get_settings
from typing import List, Dict


def compare(project1: str, project2: str, features: List[str], feature_params: Dict[str, str]) -> ComparisonResult:
    settings = get_settings()
    settings.apply_feature_params(feature_params)
    cache_filenames = settings.get_feature_basenames(features)

    project1 = load_project(project1)
    project2 = load_project(project2)
    project1, project2 = validate_project_alignment(project1, project2)

    output = ComparisonResult(
        project1=project1.name,
        project2=project2.name,
        features=features,
    )

    for feature_id, cache_filename in zip(features, cache_filenames):
        if is_reference_aware_feature(feature_id):
            result1 = compute_reference_aware_feature(feature_id, project1, project2, cache_filename)
            result2 = compute_reference_aware_feature(feature_id, project2, project1, cache_filename)
        else:
            result1 = compute_reference_free_feature(feature_id, project1, cache_filename)
            result2 = compute_reference_free_feature(feature_id, project2, cache_filename)

        if result1 and result2:
            output.results[feature_id] = FeatureComparisonResult([result1, result2])

    return output
