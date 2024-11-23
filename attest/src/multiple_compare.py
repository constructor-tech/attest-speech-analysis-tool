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
    is_non_reference_aware_metric,
    is_reference_aware_feature,
    compute_non_reference_aware_feature,
    compute_reference_aware_feature,
)
from attest.src.model import (
    load_project,
    MultipleComparisonResult,
    FeatureComparisonResult,
)
from attest.src.settings import get_settings
from typing import List, Dict


def multiple_compare(
    projects: List[str], features: List[str], feature_params: Dict[str, str]
) -> MultipleComparisonResult:
    settings = get_settings()
    settings.apply_feature_params(feature_params)
    cache_filenames = settings.get_feature_basenames(features)

    projects = [load_project(project_id) for project_id in projects]
    output = MultipleComparisonResult(projects=[ds.name for ds in projects], features=features)

    for feature_id, cache_filename in zip(features, cache_filenames):
        if not (is_reference_aware_feature(feature_id) or is_non_reference_aware_metric(feature_id)):
            continue

        feature_results = []

        for project in projects:
            result = (
                compute_reference_aware_feature(feature_id, project, projects[0], cache_filename)
                if is_reference_aware_feature(feature_id)
                else compute_non_reference_aware_feature(feature_id, project, cache_filename)
            )  # metric
            feature_results.append(result)

        output.results[feature_id] = FeatureComparisonResult(feature_results)

    return output
