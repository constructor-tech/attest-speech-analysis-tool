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

import streamlit as st
from attest.src import evaluate, compare, multiple_compare
import attest.ui.view_constants as vc
from attest.ui.model import (
    UIErrorResult,
    UIEvaluationResult,
    UIComparisonResult,
    UIMultipleComparisonResult,
)
from attest.ui.settings import get_settings
from attest.ui.view import View
from attest.ui.utils import get_logger


def init_session_state():
    if "tab" not in st.session_state:
        st.session_state.tab = vc.SETTINGS_TAB

    if "project_id" not in st.session_state:
        st.session_state.project_id = None

    if "features" not in st.session_state:
        st.session_state.features = []

    if "feature_params" not in st.session_state:
        st.session_state.feature_params = {}

    if "result" not in st.session_state:
        st.session_state.result = None

    if "parsed_result" not in st.session_state:
        st.session_state.parsed_result = None

    if "method" not in st.session_state:
        st.session_state.method = vc.EVALUATE_METHOD

    if "selected_method" not in st.session_state:
        st.session_state.selected_method = st.session_state.method

    if "group" not in st.session_state:
        st.session_state.group = vc.EMPTY_GROUP_LABEL

    if "selected_group" not in st.session_state:
        st.session_state.selected_group = st.session_state.group

    if "num_projects" not in st.session_state:
        st.session_state.num_projects = 0

    if "selected_num_projects" not in st.session_state:
        st.session_state.selected_num_projects = st.session_state.num_projects

    for i in range(st.session_state.num_projects):
        if f"selected_project_{i}" not in st.session_state:
            st.session_state[f"selected_project_{i}"] = st.session_state[f"project_{i}"]

    if "pitch_extract_method" not in st.session_state:
        st.session_state.pitch_extract_method = "parselmouth"

    if "selected_pitch_extract_method" not in st.session_state:
        st.session_state.selected_pitch_extract_method = st.session_state.pitch_extract_method

    if "text_norm_method" not in st.session_state:
        st.session_state.text_norm_method = "None"

    if "selected_text_norm_method" not in st.session_state:
        st.session_state.selected_text_norm_method = st.session_state.text_norm_method

    if "phonemization_method" not in st.session_state:
        st.session_state.phonemization_method = "openphonemizer"

    if "selected_phonemization_method" not in st.session_state:
        st.session_state.selected_phonemization_method = st.session_state.phonemization_method


def get_projects():
    projects = []
    for i in range(st.session_state.num_projects):
        if st.session_state.group == vc.EMPTY_GROUP_LABEL:
            projects.append(st.session_state[f"project_{i}"])
        else:
            project_name = st.session_state[f"project_{i}"]
            projects.append(f"{st.session_state.group}/{project_name}")

    return projects


if __name__ == "__main__":
    logger = get_logger()
    settings = get_settings()

    init_session_state()

    view = View()

    projects = get_projects()
    current_project_id = ".".join([x for x in projects if x])
    feature_params = {
        "pitch_extraction_method": st.session_state.pitch_extract_method,
        "text_normalization_method": st.session_state.text_norm_method,
        "phonemization_method": st.session_state.phonemization_method,
    }

    has_new_features = any(x not in st.session_state.features for x in settings.FEATURES)
    has_new_feature_params = any(
        st.session_state.feature_params[x] != feature_params[x] for x in st.session_state.feature_params
    )
    project_updated = st.session_state.project_id != current_project_id or has_new_features or has_new_feature_params

    if project_updated and st.session_state.tab == vc.HOME_TAB:
        try:
            if st.session_state.method == vc.EVALUATE_METHOD:
                st.session_state.result = evaluate(
                    project=projects[0],
                    features=settings.FEATURES,
                    feature_params=feature_params,
                )
                st.session_state.parsed_result = UIEvaluationResult.parse(st.session_state.result)

            elif st.session_state.method == vc.COMPARE_METHOD:
                st.session_state.result = compare(
                    project1=projects[0],
                    project2=projects[1],
                    features=settings.FEATURES,
                    feature_params=feature_params,
                )
                st.session_state.parsed_result = UIComparisonResult.parse(st.session_state.result)

            elif st.session_state.method == vc.COMPARE_MULTIPLE_METHOD:
                st.session_state.result = multiple_compare(
                    projects=projects,
                    features=settings.FEATURES,
                    feature_params=feature_params,
                )
                st.session_state.parsed_result = UIMultipleComparisonResult.parse(st.session_state.result)

            st.session_state.project_id = current_project_id
            st.session_state.features = settings.FEATURES
            st.session_state.feature_params = feature_params

        except FileNotFoundError as e:
            error_msg = vc.FILE_NOT_FOUND(e)
            st.session_state.parsed_result = UIErrorResult(error_msg)

    if st.session_state.tab == vc.SETTINGS_TAB:
        view.display_settings()

    elif isinstance(st.session_state.parsed_result, UIErrorResult):
        view.display_error(st.session_state.parsed_result.msg)

    elif isinstance(st.session_state.parsed_result, UIEvaluationResult):
        view.display_evaluation_result()

    elif isinstance(st.session_state.parsed_result, UIComparisonResult):
        view.display_comparison_result()

    elif isinstance(st.session_state.parsed_result, UIMultipleComparisonResult):
        view.display_multi_comparison_result()
