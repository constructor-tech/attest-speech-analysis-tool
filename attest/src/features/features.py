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

from typing import Union

from attest.src.settings import get_settings
from attest.src.model import Project, AttributeResult, MetricResult
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import (
    validate_feature_from_cache,
    validate_reference_aware_feature_from_cache,
)
from attest.src.utils.logger import get_logger
from attest.src.utils.performance_tracker import PerformanceTracker
from .attributes import get_attribute_id_to_method
from .metrics import get_metric_id_to_method
from .reference_aware_metrics import get_reference_aware_metric_id_to_method


logger = get_logger()
settings = get_settings()


_attribute_id_to_method = get_attribute_id_to_method()
_reference_aware_metric_id_to_method = get_reference_aware_metric_id_to_method()
_reference_free_metric_id_to_method = get_metric_id_to_method()
_metric_id_to_method = {
    **_reference_aware_metric_id_to_method,
    **_reference_free_metric_id_to_method,
}
_reference_aware_feature_id_to_method = _reference_aware_metric_id_to_method
_reference_free_feature_id_to_method = {
    **_attribute_id_to_method,
    **_reference_free_metric_id_to_method,
}
_feature_id_to_method = {
    **_attribute_id_to_method,
    **_reference_aware_metric_id_to_method,
    **_reference_free_metric_id_to_method,
}


def is_attribute(feature_id: str) -> bool:
    return feature_id in _attribute_id_to_method


def is_metric(feature_id: str) -> bool:
    return feature_id in _metric_id_to_method


def is_reference_aware_metric(feature_id: str) -> bool:
    return feature_id in _reference_aware_metric_id_to_method


def is_reference_free_metric(feature_id: str) -> bool:
    return feature_id in _reference_free_metric_id_to_method


def is_feature(feature_id: str) -> bool:
    return feature_id in _feature_id_to_method


def is_reference_aware_feature(feature_id: str) -> bool:
    return feature_id in _reference_aware_feature_id_to_method


def is_reference_free_feature(feature_id: str) -> bool:
    return feature_id in _reference_free_feature_id_to_method


@CacheHandler(
    cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/features1/${{2}}.pickle",
    method="pickle",
    validator=validate_feature_from_cache,
)
def compute_reference_free_feature(
    feature_id: str, project: Project, cache_filename: str
) -> Union[AttributeResult, MetricResult]:
    tracker = PerformanceTracker(name=f'Computing feature "{feature_id}" for project "{project.name}"', start=True)

    feature = _reference_free_feature_id_to_method.get(feature_id)
    if feature is None:
        logger.info('Unknown feature "%s", skipping.' % feature_id)
        return None

    feature_result = feature(project)

    tracker.end()
    return feature_result


@CacheHandler(
    cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/features2/${{2.name}}/${{3}}.pickle",
    method="pickle",
    validator=validate_reference_aware_feature_from_cache,
)
def compute_reference_aware_feature(
    feature_id: str, hyp_project: Project, ref_project: Project, cache_filename: str
) -> Union[AttributeResult, MetricResult]:
    print(1, settings.PITCH_EXTRACT_METHOD)
    tracker = PerformanceTracker(
        name=f'Computing feature "{feature_id}" for projects "{hyp_project.name}" (hyp), "{ref_project.name}" (ref)',
        start=True,
    )

    feature = _reference_aware_feature_id_to_method.get(feature_id)
    if feature is None:
        logger.info('Unknown feature "%s", skipping.' % feature_id)
        return None

    feature_result = feature(hyp_project, ref_project)

    tracker.end()
    return feature_result
