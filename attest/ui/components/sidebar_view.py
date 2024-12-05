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
import attest.ui.constants as vc
from attest.ui.model import MetricType
from attest.ui.settings import Settings
from attest.ui.utils.view_utils import (
    toggle_features,
    get_project_name,
)


class SidebarView:

    def __init__(self, settings: Settings):
        self.settings = settings

    def setup_sidebar(self):
        st.sidebar.subheader(vc.FEATURES_LABEL)

        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.button("Enable all", on_click=lambda: toggle_features(self.settings, enable=True))
        with col2:
            st.button("Disable all", on_click=lambda: toggle_features(self.settings, enable=False))

        selected_features = []
        for cat, cat_features in self.settings.ALL_FEATURES:
            with st.sidebar.expander(cat):
                for label, feature, feature_type, _ in cat_features:
                    disabled = (
                        feature_type == MetricType.REFERENCE_AWARE_METRIC
                        and st.session_state.selected_method == vc.EVALUATE_METHOD
                    )
                    selected = st.checkbox(
                        label,
                        value=(feature in self.settings.FEATURES and not disabled),
                        disabled=disabled,
                    )
                    if selected:
                        selected_features.append(feature)
        self.settings.FEATURES = selected_features

    def display_sidebar(self):
        """Display the sidebar based on the current tab and method."""
        if st.session_state.tab != vc.HOME_TAB:
            return

        if st.session_state.method not in [vc.EVALUATE_METHOD, vc.COMPARE_METHOD]:
            return

        self.display_sorting_controls()
        if st.session_state.method == vc.EVALUATE_METHOD:
            self.display_filtering_controls()
        if self.settings.DISPLAY_DETAILED_SECTION:
            self.display_navigation_controls()

    def display_sorting_controls(self):
        """Display sorting options in the sidebar."""
        metric_keys = [key for key in self.settings.FEATURES if key in st.session_state.parsed_result.overall_metrics]
        sorting_options = [vc.SORT_AND_FILTER_NONE_LABEL, *metric_keys]

        st.sidebar.subheader(vc.SORT_LABEL)
        st.sidebar.selectbox(
            vc.SORT_LABEL,
            sorting_options,
            label_visibility="collapsed",
            key="sort_option",
        )

        if st.session_state.method == vc.COMPARE_METHOD:
            project1_name = get_project_name(st.session_state.parsed_result.project1)
            project2_name = get_project_name(st.session_state.parsed_result.project2)
            st.sidebar.selectbox(
                vc.SORT_METHOD_LABEL,
                [vc.SORT_BY_DIFFERENCE_LABEL, project1_name, project2_name],
                key="sort_method",
            )

    def display_filtering_controls(self):
        """Display filtering options in the sidebar."""
        metric_keys = [key for key in self.settings.FEATURES if key in st.session_state.parsed_result.overall_metrics]
        filtering_options = [vc.SORT_AND_FILTER_NONE_LABEL, *metric_keys]

        st.sidebar.subheader(vc.FILTER_LABEL)
        st.sidebar.selectbox(
            vc.FILTER_LABEL,
            filtering_options,
            label_visibility="collapsed",
            key="filter_option",
        )

        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            self.min_filter_value = st.text_input(vc.FILTER_MIN_VALUE_LABEL, "")
        with col2:
            self.max_filter_value = st.text_input(vc.FILTER_MAX_VALUE_LABEL, "")

    def display_navigation_controls(self):
        """Display navigation controls in the sidebar."""
        st.sidebar.subheader(vc.NAVIGATION_LABEL)
        total_pages = (
            st.session_state.num_detailed_elements + self.settings.ITEMS_PER_PAGE - 1
        ) // self.settings.ITEMS_PER_PAGE
        self.current_page = st.sidebar.number_input(
            vc.NAVIGATION_PAGE_LABEL, min_value=1, value=1, max_value=total_pages
        )
        st.sidebar.write(vc.NAVIGATION_TOTAL_PAGES_TEXT(total_pages))
