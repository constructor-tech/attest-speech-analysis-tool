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

import json
import streamlit as st

import attest.ui.constants as vc
from attest.ui.settings import Settings
from attest.ui.utils.configuration_utils import (
    check_if_group,
    get_list_of_groups,
    get_list_of_projects,
    get_list_of_pitch_extract_methods,
    get_list_of_text_norm_methods,
    get_list_of_phonemization_methods,
    get_list_of_languages_whisper,
    get_list_of_languages_espeak,
    resolve_group_path,
)


class ConfigurationView:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.export_filename = "Metrics.json"

    def display_settings(self):
        st.subheader(vc.CONFIGURATION_TAB)
        self.display_export_subsection()
        self.display_method_subsection()
        self.display_group_and_project_subsection()
        self.display_feature_configuration()
        self.display_view_configuration()

    def display_export_subsection(self):
        st.write(vc.EXPORT_LABEL)
        st.download_button(
            label=vc.EXPORT_BUTTON_LABEL,
            data=json.dumps(
                st.session_state.result.to_dict() if st.session_state.result else {},
                indent=4,
            ),
            file_name=self.export_filename,
            mime="application/json",
        )

    def display_method_subsection(self):
        st.write(vc.METHOD_LABEL)

        values = [vc.EVALUATE_METHOD, vc.COMPARE_METHOD, vc.COMPARE_MULTIPLE_METHOD]
        default_index = values.index(st.session_state.selected_method)

        st.selectbox(
            vc.METHOD_LABEL,
            values,
            label_visibility="collapsed",
            index=default_index,
            key="selected_method",
        )

    def display_group_and_project_subsection(self):
        st.write(vc.GROUP_AND_PROJECT_LABEL)
        selected_group = st.session_state.selected_group
        selected_method = st.session_state.selected_method

        # Display groups
        group_identifiers = []

        if check_if_group(self.settings.DATA_DIR):
            group_identifiers.append(vc.EMPTY_GROUP_LABEL)

        list_of_groups = get_list_of_groups(self.settings.DATA_DIR)

        group_identifiers.extend(list_of_groups)

        group_index = 0
        if selected_group in group_identifiers:
            group_index = group_identifiers.index(selected_group)
        elif len(group_identifiers) > 0:
            selected_group = group_identifiers[group_index]
            st.session_state.selected_group = group_identifiers[group_index]

        st.selectbox(
            vc.GROUP_LABEL,
            group_identifiers,
            index=group_index,
            key="selected_group",
        )

        # Display projects
        if selected_group == vc.EMPTY_GROUP_LABEL:
            selected_group = None

        group_path = resolve_group_path(self.settings.DATA_DIR, selected_group)

        list_of_projects = sorted(get_list_of_projects(group_path))

        if selected_method == vc.EVALUATE_METHOD:
            st.session_state.selected_num_projects = 1
            self.projects = [st.selectbox(vc.PROJECT_LABEL, list_of_projects, key="selected_project_0")]

        elif selected_method == vc.COMPARE_METHOD:
            st.session_state.selected_num_projects = 2
            second_index = min(1, len(list_of_projects) - 1)
            self.projects = [
                st.selectbox(vc.PROJECT_X_LABEL(1), list_of_projects, key="selected_project_0"),
                st.selectbox(
                    vc.PROJECT_X_LABEL(2),
                    list_of_projects,
                    index=second_index,
                    key="selected_project_1",
                ),
            ]

        elif selected_method == vc.COMPARE_MULTIPLE_METHOD:
            self.settings.NUM_PROJECTS_TO_COMPARE = st.text_input(
                vc.NUM_PROJECTS_FOR_MULTIPLE_COMPARE_LABEL,
                value=self.settings.NUM_PROJECTS_TO_COMPARE,
            )
            st.session_state.selected_num_projects = int(self.settings.NUM_PROJECTS_TO_COMPARE)
            self.projects = []
            for i in range(int(self.settings.NUM_PROJECTS_TO_COMPARE)):
                index = min(i, len(list_of_projects) - 1)
                self.projects.append(
                    st.selectbox(
                        vc.PROJECT_X_LABEL(i + 1),
                        list_of_projects,
                        index=index,
                        key=f"selected_project_{i}",
                    )
                )

    def display_feature_configuration(self):
        st.write(vc.FEATURE_CONFIGURATION_LABEL)

        pitch_extract_methods = get_list_of_pitch_extract_methods()
        index = 0
        for i, method in enumerate(pitch_extract_methods):
            if method == st.session_state.pitch_extract_method:
                index = i
        st.selectbox(
            vc.PITCH_EXTRACT_METHOD_LABEL,
            pitch_extract_methods,
            index=index,
            key="selected_pitch_extract_method",
        )

        text_norm_methods = get_list_of_text_norm_methods()
        index = 0
        for i, method in enumerate(text_norm_methods):
            if method == st.session_state.text_norm_method:
                index = i
        st.selectbox(
            vc.TEXT_NORM_METHOD_LABEL,
            text_norm_methods,
            index=index,
            key="selected_text_norm_method",
        )

        phonemization_methods = get_list_of_phonemization_methods()
        index = 0
        for i, method in enumerate(phonemization_methods):
            if method == st.session_state.phonemization_method:
                index = i
        st.selectbox(
            vc.PHONEMIZATION_METHOD_LABEL,
            phonemization_methods,
            index=index,
            key="selected_phonemization_method",
        )

        wshiper_languages = get_list_of_languages_whisper()
        index = 0
        for i, method in enumerate(wshiper_languages):
            if method == st.session_state.whisper_language:
                index = i
        st.selectbox(
            vc.WHISPER_LANGUAGE_LABEL,
            wshiper_languages,
            index=index,
            key="selected_whisper_language",
        )

        espeak_languages = get_list_of_languages_espeak()
        espeak_lang_selecbox_disabled = st.session_state.selected_phonemization_method != "espeak_phonemizer"
        index = 0
        for i, method in enumerate(espeak_languages):
            if method == st.session_state.espeak_language:
                index = i
        st.selectbox(
            vc.ESPEAK_LANGUAGE_LABEL,
            espeak_languages,
            index=index,
            key="selected_espeak_language",
            disabled=espeak_lang_selecbox_disabled,
        )

    def display_view_configuration(self):
        show_analysis_conf = st.session_state.selected_method == vc.EVALUATE_METHOD
        show_summary_conf = st.session_state.selected_method == vc.EVALUATE_METHOD
        show_detailed_conf = (
            st.session_state.selected_method == vc.EVALUATE_METHOD
            or st.session_state.selected_method == vc.COMPARE_METHOD
        )
        show_comparison_plots_conf = st.session_state.selected_method == vc.COMPARE_MULTIPLE_METHOD
        show_first_column_conf = st.session_state.selected_method == vc.COMPARE_MULTIPLE_METHOD

        st.write(vc.VIEW_CONFIGURATION_LABEL)

        if show_analysis_conf:
            self.settings.DISPLAY_ANALYSIS_SECTOIN = st.checkbox(
                label=vc.DISPLAY_ANALYSIS_LABEL,
                value=self.settings.DISPLAY_ANALYSIS_SECTOIN,
            )

        if show_summary_conf:
            self.settings.DISPLAY_SUMMARY_SECTOIN = st.checkbox(
                label=vc.DISPLAY_SUMMARY_LABEL,
                value=self.settings.DISPLAY_SUMMARY_SECTOIN,
            )
            self.settings.SUMMARY_NUM_COLS = int(
                st.text_input(
                    label=vc.ANALYSIS_NUM_COLUMNS_LABEL,
                    value=f"{self.settings.SUMMARY_NUM_COLS}",
                    disabled=not self.settings.DISPLAY_SUMMARY_SECTOIN,
                )
            )
            self.settings.SUMMARY_SUBSET_INDEX = int(
                st.text_input(
                    label=vc.SUMMARY_SUBSET_INDEX_LABEL,
                    value=f"{self.settings.SUMMARY_SUBSET_INDEX}",
                    disabled=not self.settings.DISPLAY_SUMMARY_SECTOIN,
                )
            )

        if show_detailed_conf:
            self.settings.DISPLAY_DETAILED_SECTION = st.checkbox(
                label=vc.DISPLAY_DETAILED_LABEL,
                value=self.settings.DISPLAY_DETAILED_SECTION,
            )
            self.settings.ITEMS_PER_PAGE = int(
                st.text_input(
                    label=vc.DETAILED_NUM_ITEMS_LABEL,
                    value=f"{self.settings.ITEMS_PER_PAGE}",
                    disabled=not self.settings.DISPLAY_DETAILED_SECTION,
                ),
            )

        if show_first_column_conf:
            self.settings.SHOW_FIRST_COLUMN = st.checkbox(
                label=vc.SHOW_FIRST_COLUMN_LABEL,
                value=self.settings.SHOW_FIRST_COLUMN,
            )

        if show_first_column_conf:
            self.settings.TRANSPOSE_OVERALL_TABLE = st.checkbox(
                label=vc.TRANSPOSE_OVERALL_TABLE_LABEL,
                value=self.settings.TRANSPOSE_OVERALL_TABLE,
            )

        if show_comparison_plots_conf:
            self.settings.DISPLAY_COMPARISON_PLOTS = st.checkbox(
                label=vc.DISPLAY_COMPARISON_PLOTS_LABEL,
                value=self.settings.DISPLAY_COMPARISON_PLOTS,
            )

        if show_comparison_plots_conf:
            self.settings.TRANSPOSE_COMPARISON_PLOTS = st.checkbox(
                label=vc.TRANSPOSE_COMPARISON_PLOTS_LABEL,
                value=self.settings.TRANSPOSE_COMPARISON_PLOTS,
                disabled=not self.settings.DISPLAY_COMPARISON_PLOTS,
            )
