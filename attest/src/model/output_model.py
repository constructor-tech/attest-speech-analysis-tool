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

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Union

NumericType = Union[int, float]


@dataclass
class AttributeResultEntry:
    uid: str
    message: str = field(default="")
    audio_path: str = field(default="")
    image_path: str = field(default="")
    video_path: str = field(default="")
    plot_data: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class AttributeResult:
    detailed: List[AttributeResultEntry]

    def to_dict(self):
        return {"detailed": [entry.to_dict() for entry in self.detailed]}


@dataclass
class MetricResultEntry:
    uid: str
    score: NumericType

    def to_dict(self):
        return asdict(self)


@dataclass
class MetricResult:
    overall: NumericType
    detailed: List[MetricResultEntry]

    def to_dict(self):
        return {
            "overall": self.overall,
            "detailed": [entry.to_dict() for entry in self.detailed],
        }


@dataclass
class FeatureComparisonResult:
    results: List[Union[AttributeResult, MetricResult]]

    def to_dict(self):
        return {"results": [result.to_dict() for result in self.results]}


@dataclass
class EvaluationResult:
    project: str
    features: List[str]
    results: Dict[str, Union[AttributeResult, MetricResult]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "project": self.project,
            "features": self.features,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


@dataclass
class ComparisonResult:
    project1: str
    project2: str
    features: List[str]
    results: Dict[str, Union[AttributeResult, MetricResult, FeatureComparisonResult]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "project1": self.project1,
            "project2": self.project2,
            "features": self.features,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


@dataclass
class MultipleComparisonResult:
    projects: List[str]
    features: List[str]
    results: Dict[str, Union[AttributeResult, MetricResult, FeatureComparisonResult]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "projects": self.projects,
            "features": self.features,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }
