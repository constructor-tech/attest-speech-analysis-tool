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

from enum import Enum
from attest.src.model import (
    EvaluationResult,
    ComparisonResult,
    MultipleComparisonResult,
    MetricResult,
    FeatureComparisonResult,
)


class MetricType(Enum):
    METRIC = "Metric"
    REFERENCE_AWARE_METRIC = "ReferenceAwareMetric"
    ATTRIBUTE = "Attribute"


class UIErrorResult:
    def __init__(self, msg):
        self.msg = msg


class UIEvaluationResult:

    def __init__(self, project, overall_metrics, detailed_features):
        self.project = project
        self.overall_metrics = overall_metrics
        self.detailed_features = detailed_features

    @classmethod
    def parse(cls, result: EvaluationResult):
        overall_metrics = {}
        detailed_features = {}

        for feature_id, feature_result in result.results.items():

            if isinstance(feature_result, MetricResult):
                overall_metrics[feature_id] = feature_result.overall

            for entry in feature_result.detailed:
                if entry.uid not in detailed_features:
                    detailed_features[entry.uid] = {}
                detailed_features[entry.uid][feature_id] = entry

        return cls(
            project=result.project,
            overall_metrics=overall_metrics,
            detailed_features=detailed_features,
        )


class UIComparisonResult:

    def __init__(self, project1, project2, overall_metrics, detailed_features):
        self.project1 = project1
        self.project2 = project2
        self.overall_metrics = overall_metrics
        self.detailed_features = detailed_features

    @classmethod
    def parse(cls, result: ComparisonResult):
        overall_metrics = {}
        detailed_features = {}

        for feature_id, feature_result in result.results.items():

            if isinstance(feature_result, FeatureComparisonResult):
                feature_result1 = feature_result.results[0]
                feature_result2 = feature_result.results[1]

                if isinstance(feature_result1, MetricResult) and isinstance(feature_result2, MetricResult):
                    overall_metrics[feature_id] = (
                        feature_result1.overall,
                        feature_result2.overall,
                    )

                for entry1 in feature_result1.detailed:
                    for entry2 in feature_result2.detailed:
                        # TODO @od 26.07.2024: rewrite with O(n*logn) time complexity
                        if entry1.uid == entry2.uid:
                            if entry1.uid not in detailed_features:
                                detailed_features[entry1.uid] = {}
                            detailed_features[entry1.uid][feature_id] = (entry1, entry2)

            else:
                if isinstance(feature_result, MetricResult):
                    overall_metrics[feature_id] = feature_result.overall

                for entry in feature_result.detailed:
                    if entry.uid not in detailed_features:
                        detailed_features[entry.uid] = {}
                    detailed_features[entry.uid][feature_id] = entry

        return cls(
            project1=result.project1,
            project2=result.project2,
            overall_metrics=overall_metrics,
            detailed_features=detailed_features,
        )


class UIMultipleComparisonResult:

    def __init__(self, projects, overall_metrics):
        self.projects = projects
        self.overall_metrics = overall_metrics

    @classmethod
    def parse(cls, result: MultipleComparisonResult):
        overall_metrics = {}

        for feature_id, feature_result in result.results.items():
            if isinstance(feature_result, FeatureComparisonResult):
                for project_id, metric_result in zip(result.projects, feature_result.results):
                    if isinstance(metric_result, MetricResult):
                        if feature_id not in overall_metrics:
                            overall_metrics[feature_id] = {}
                        overall_metrics[feature_id][project_id] = metric_result.overall

        return cls(
            projects=result.projects,
            overall_metrics=overall_metrics,
        )
