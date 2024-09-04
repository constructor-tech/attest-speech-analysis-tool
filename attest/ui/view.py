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
import matplotlib.pyplot as plt
import os
import streamlit as st

import attest.ui.view_constants as vc

from PIL import Image
from streamlit_option_menu import option_menu

from attest.ui.model import (
    UIEvaluationResult,
    UIComparisonResult,
    UIMultipleComparisonResult,
    MetricType,
)
from attest.ui.settings import get_settings
from attest.ui.utils import (
    get_list_of_projects,
    get_list_of_pitch_extract_methods,
    get_list_of_text_norm_methods,
)
from attest.ui.view_utils import (
    convert_to_dataframe,
    convert_to_table,
    dataframe_to_latex,
    dataframe_to_markdown,
    display_boxplot,
    display_correlation_plot,
    display_plot,
)


class View:
    def __init__(self):
        self.settings = get_settings()

        self.export_filename = "Metrics.json"

        self.projects = []

        favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
        st.set_page_config(page_title=vc.PROJECT_TITLE, page_icon=favicon_path, layout="wide")

        if "num_detailed_elements" not in st.session_state:
            st.session_state.num_detailed_elements = 1

        self.setup_menu()
        self.setup_sidebar()

    def setup_menu(self):
        def on_tab_change(_):
            if st.session_state.tab == vc.HOME_TAB:
                st.session_state.method = st.session_state.selected_method
                st.session_state.group = st.session_state.selected_group
                st.session_state.num_projects = st.session_state.selected_num_projects
                for i in range(st.session_state.num_projects):
                    st.session_state[f"project_{i}"] = st.session_state[f"selected_project_{i}"]
                st.session_state.pitch_extract_method = st.session_state.selected_pitch_extract_method
                st.session_state.text_norm_method = st.session_state.selected_text_norm_method

        option_menu(
            None,
            [vc.HOME_TAB, vc.SETTINGS_TAB],
            icons=["house", "gear"],
            menu_icon="cast",
            default_index=1,
            orientation="horizontal",
            key="tab",
            on_change=on_tab_change,
        )

    def setup_sidebar(self):
        st.sidebar.subheader(vc.FEATURES_LABEL)

        def on_click_enable():
            self.settings.FEATURES = [
                feature for (_, cat_features) in self.settings.ALL_FEATURES for (_, feature, _, _) in cat_features
            ]

        def on_click_disable():
            self.settings.FEATURES = []

        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.button("Enable all", on_click=on_click_enable)
        with col2:
            st.button("Disable all", on_click=on_click_disable)

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
                        value=feature in self.settings.FEATURES,
                        disabled=disabled,
                    )
                    if selected:
                        selected_features.append(feature)
        self.settings.FEATURES = selected_features

    def _get_project_name(self, project):
        return (
            project
            if st.session_state.group == vc.EMPTY_GROUP_LABEL
            else project.replace(f"{st.session_state.group}/", "")
        )

    def display_settings(self):
        st.subheader(vc.SETTINGS_TAB)
        self.display_export_subsection()
        self.display_method_subsection()
        self.display_group_and_project_subsection()
        self.display_feature_configuration()
        self.display_view_configuration()

    def display_error(self, msg):
        st.write(msg)

    def display_sidebar(self):
        if st.session_state.tab != vc.HOME_TAB:
            return

        if st.session_state.method != vc.EVALUATE_METHOD and st.session_state.method != vc.COMPARE_METHOD:
            return

        metric_keys = [key for key in self.settings.FEATURES if key in st.session_state.parsed_result.overall_metrics]
        sorting_and_filtering_options = [vc.SORT_AND_FILTER_NONE_LABEL, *metric_keys]

        st.sidebar.subheader(vc.SORT_LABEL)
        st.sidebar.selectbox(
            vc.SORT_LABEL,
            sorting_and_filtering_options,
            label_visibility="collapsed",
            key="sort_option",
        )

        if st.session_state.method == vc.COMPARE_METHOD:
            project1_name = self._get_project_name(st.session_state.parsed_result.project1)
            project2_name = self._get_project_name(st.session_state.parsed_result.project2)
            st.sidebar.selectbox(
                vc.SORT_METHOD_LABEL,
                [vc.SORT_BY_DIFFERENCE_LABEL, project1_name, project2_name],
                key="sort_method",
            )

        if st.session_state.method == vc.EVALUATE_METHOD:
            st.sidebar.subheader(vc.FILTER_LABEL)
            st.sidebar.selectbox(
                vc.FILTER_LABEL,
                sorting_and_filtering_options,
                label_visibility="collapsed",
                key="filter_option",
            )

            col1, col2 = st.sidebar.columns([1, 1])
            with col1:
                self.min_filter_value = st.text_input(vc.FILTER_MIN_VALUE_LABEL, "")
            with col2:
                self.max_filter_value = st.text_input(vc.FILTER_MAX_VALUE_LABEL, "")

        if self.settings.DISPLAY_DETAILED_SECTION:
            st.sidebar.subheader(vc.NAVIGATION_LABEL)
            total_pages = (
                st.session_state.num_detailed_elements + self.settings.ITEMS_PER_PAGE - 1
            ) // self.settings.ITEMS_PER_PAGE
            self.current_page = st.sidebar.number_input(
                vc.NAVIGATION_PAGE_LABEL, min_value=1, value=1, max_value=total_pages
            )
            st.sidebar.write(vc.NAVIGATION_TOTAL_PAGES_TEXT(total_pages))

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

        group_values = [vc.EMPTY_GROUP_LABEL, *os.listdir(self.settings.DATA_DIR)]
        group_default_index = group_values.index(st.session_state.selected_group)
        st.selectbox(
            vc.GROUP_LABEL,
            group_values,
            index=group_default_index,
            key="selected_group",
        )

        list_of_projects = get_list_of_projects(st.session_state.selected_group)

        if st.session_state.selected_method == vc.EVALUATE_METHOD:
            st.session_state.selected_num_projects = 1
            self.projects = [st.selectbox(vc.PROJECT_LABEL, list_of_projects, key="selected_project_0")]

        elif st.session_state.selected_method == vc.COMPARE_METHOD:
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

        elif st.session_state.selected_method == vc.COMPARE_MULTIPLE_METHOD:
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

    def display_evaluation_result(self):
        self.display_sidebar()

        result: UIEvaluationResult = st.session_state.parsed_result

        metric_keys = [key for key in self.settings.FEATURES if key in result.overall_metrics]
        detailed_list = list(result.detailed_features.values())

        # Main field
        project_name = self._get_project_name(result.project)
        st.subheader(f"Project: {project_name}")

        # Overall section
        st.subheader(vc.OVERALL_LABEL)
        for metric_id, overall_score in result.overall_metrics.items():
            if metric_id in self.settings.FEATURES:
                st.write(self.settings.FEATURE_ID_TO_LABEL[metric_id], overall_score)

        # Analysis section
        if self.settings.DISPLAY_ANALYSIS_SECTOIN:
            st.subheader(vc.ANALYSIS_LABEL)
            metric_labels = [self.settings.FEATURE_ID_TO_LABEL[x] for x in metric_keys]
            feature1_option = st.selectbox(vc.ANALYSIS_FIRST_FEATURE_LABEL, metric_labels)
            feature2_option = st.selectbox(vc.ANALYSIS_SECOND_FEATURE_LABEL, metric_labels)
            feature1_key = self.settings.FEATURE_LABEL_TO_ID[feature1_option]
            feature2_key = self.settings.FEATURE_LABEL_TO_ID[feature2_option]
            feature1_values = [x[feature1_key].score for x in detailed_list]
            feature2_values = [x[feature2_key].score for x in detailed_list]
            display_correlation_plot(feature1_option, feature2_option, feature1_values, feature2_values)

        # Summary section
        if self.settings.DISPLAY_SUMMARY_SECTOIN:
            st.subheader(vc.SUMMARY_LABEL)
            if self.settings.SUMMARY_SUBSET_INDEX < 0:
                subsets = []
            else:
                subsets = set([k.split("/")[self.settings.SUMMARY_SUBSET_INDEX] for k in result.detailed_features])
            cols = st.columns(self.settings.SUMMARY_NUM_COLS)

            for i, (metric_id, overall_score) in enumerate(result.overall_metrics.items()):
                if metric_id not in self.settings.FEATURES:
                    continue
                boxplot_data = [entry[metric_id].score for entry in detailed_list]
                boxplot_data = [boxplot_data]

                if len(subsets) > 1:
                    labels = ["all"]
                    for subset in subsets:
                        # TODO @od 2024.08.-5: O(subsets * result.detailed_features) -> O(result.detailed_features)
                        subset_data = [
                            entry[metric_id].score
                            for k, entry in result.detailed_features.items()
                            if k.split("/")[self.settings.SUMMARY_SUBSET_INDEX] == subset
                        ]
                        boxplot_data.append(subset_data)
                        labels.append(subset)
                else:
                    labels = None

                figsize = (16, 8) if len(boxplot_data) > 3 else (8, 4)
                with cols[i % self.settings.SUMMARY_NUM_COLS]:
                    display_boxplot(
                        cols[i % self.settings.SUMMARY_NUM_COLS],
                        boxplot_data,
                        labels,
                        figsize=figsize,
                        title=self.settings.FEATURE_ID_TO_LABEL[metric_id],
                    )

        # Detailed section
        if self.settings.DISPLAY_DETAILED_SECTION:
            st.subheader(vc.DETAILED_LABEL)

            # Filter the detailed_list
            if st.session_state.filter_option != vc.SORT_AND_FILTER_NONE_LABEL:
                min_val = float(self.min_filter_value) if self.min_filter_value != "" else -float("inf")
                max_val = float(self.max_filter_value) if self.max_filter_value != "" else float("inf")
                detailed_list = [
                    x for x in detailed_list if min_val <= x[st.session_state.filter_option].score <= max_val
                ]

            # Sort the detailed_list
            if st.session_state.sort_option != vc.SORT_AND_FILTER_NONE_LABEL:
                detailed_list = sorted(detailed_list, key=lambda x: -x[st.session_state.sort_option].score)

            # Keep number of elements for sidebar navigation
            st.session_state["num_detailed_elements"] = len(detailed_list)

            # Main field, detailed section
            start_index = (self.current_page - 1) * self.settings.ITEMS_PER_PAGE
            end_index = min(start_index + self.settings.ITEMS_PER_PAGE, len(detailed_list))

            for entry in detailed_list[start_index:end_index]:
                for feature_id in self.settings.FEATURES:
                    if feature_id not in entry:
                        continue
                    feature = entry[feature_id]
                    self.display_detailed_fetuare(
                        feature_label=self.settings.FEATURE_ID_TO_LABEL[feature_id],
                        feature=feature,
                        is_metric=feature_id in metric_keys,
                        container=st,
                    )
                st.write("")
                st.write("")

    def display_comparison_result(self):
        self.display_sidebar()

        result: UIComparisonResult = st.session_state.parsed_result

        metric_keys = [key for key in self.settings.FEATURES if key in result.overall_metrics]
        project1_name = self._get_project_name(result.project1)
        project2_name = self._get_project_name(result.project2)
        detailed_list = list(result.detailed_features.values())

        # Main field
        # Overall section
        st.subheader(vc.OVERALL_LABEL)
        overall_content = [["Metric", project1_name, project2_name]]
        for metric_id, overall_scores in result.overall_metrics.items():
            if metric_id in self.settings.FEATURES:
                overall_content.append(
                    [
                        self.settings.FEATURE_ID_TO_LABEL[metric_id],
                        overall_scores[0],
                        overall_scores[1],
                    ]
                )
        st.markdown(convert_to_table(overall_content), unsafe_allow_html=True)

        # Detailed section
        if self.settings.DISPLAY_DETAILED_SECTION:
            st.subheader(vc.DETAILED_LABEL)

            # Sort the detailed_list
            if st.session_state.sort_option != vc.SORT_AND_FILTER_NONE_LABEL:
                # Sorting Key
                if st.session_state.sort_method == project1_name:
                    entry_key = lambda x: -x[st.session_state.sort_option][0].score  # noqa: E731
                elif st.session_state.sort_method == project2_name:
                    entry_key = lambda x: -x[st.session_state.sort_option][1].score  # noqa: E731
                else:  # By Absolute Diference
                    entry_key = lambda x: -abs(  # noqa: E731
                        x[st.session_state.sort_option][0].score - x[st.session_state.sort_option][1].score
                    )

                detailed_list = sorted(detailed_list, key=entry_key)

            # Keep number of elements for sidebar navigation
            st.session_state["num_detailed_elements"] = len(detailed_list)

            # Main field, detailed section
            start_index = (self.current_page - 1) * self.settings.ITEMS_PER_PAGE
            end_index = min(start_index + self.settings.ITEMS_PER_PAGE, len(detailed_list))

            for entry in detailed_list[start_index:end_index]:
                cols = st.columns(2)
                for j in range(2):
                    for feature_id in self.settings.FEATURES:
                        if feature_id not in entry:
                            continue
                        feature = entry[feature_id][j]
                        self.display_detailed_fetuare(
                            feature_label=self.settings.FEATURE_ID_TO_LABEL[feature_id],
                            feature=feature,
                            is_metric=feature_id in metric_keys,
                            container=cols[j],
                        )
                st.write("")
                st.write("")

    def display_multi_comparison_result(self):
        result: UIMultipleComparisonResult = st.session_state.parsed_result

        project_ids = result.projects
        feature_ids = result.overall_metrics.keys()
        project_names = [self._get_project_name(x) for x in result.projects]
        if not self.settings.SHOW_FIRST_COLUMN:
            project_names = project_names[1:]

        overall_content = [["Metric", *project_names]]
        metrics_data = {}  # To store data for plotting

        for metric_id in feature_ids:
            if metric_id not in self.settings.FEATURES:
                continue
            metric_result = [result.overall_metrics[metric_id][project_id] for project_id in project_ids]
            if not self.settings.SHOW_FIRST_COLUMN:
                metric_result = metric_result[1:]
            overall_content.append([self.settings.FEATURE_ID_TO_LABEL[metric_id], *metric_result])
            metrics_data[self.settings.FEATURE_ID_TO_LABEL[metric_id]] = metric_result

        if self.settings.TRANSPOSE_OVERALL_TABLE:
            overall_content = [list(row) for row in zip(*overall_content)]

        st.subheader(vc.OVERALL_LABEL)
        st.markdown(convert_to_table(overall_content), unsafe_allow_html=True)
        st.write("")

        df = convert_to_dataframe(overall_content)
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="result.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download as Markdown",
            data=dataframe_to_markdown(df).encode("utf-8"),
            file_name="result.md",
            mime="text/markdown",
        )
        st.download_button(
            label="Download as LaTeX",
            data=dataframe_to_latex(df).encode("utf-8"),
            file_name="result.tex",
            mime="text/latex",
        )

        # Generate and display plots
        if self.settings.DISPLAY_COMPARISON_PLOTS:
            st.subheader(vc.COMPARISON_PLOTS_LABEL)
            for metric_name, values in metrics_data.items():
                fig, ax = plt.subplots()
                if self.settings.TRANSPOSE_COMPARISON_PLOTS:
                    ax.barh(project_names, values)
                else:
                    ax.bar(project_names, values)
                ax.set_title(metric_name)
                if self.settings.TRANSPOSE_COMPARISON_PLOTS:
                    ax.set_xlabel(vc.COMPARISON_PLOTS_LABEL_Y)
                    ax.set_ylabel(vc.COMPARISON_PLOTS_LABEL_X)
                else:
                    ax.set_xlabel(vc.COMPARISON_PLOTS_LABEL_X)
                    ax.set_ylabel(vc.COMPARISON_PLOTS_LABEL_Y)
                st.pyplot(fig)

    def display_detailed_fetuare(self, feature_label, feature, is_metric, container):

        if is_metric:
            # metric
            container.write(f"*{feature_label.upper()}*: {feature.score:.3f}")

        else:
            # attribute
            if feature.message:
                container.write("*%s*: %s" % (feature_label.upper(), feature.message))
            if feature.audio_path:
                container.audio(feature.audio_path)
            if feature.image_path:
                image = Image.open(feature.image_path)
                container.image(image, caption=feature.image_path)
            if feature.video_path:
                container.video(feature.video_path)
            if feature.plot_data:
                display_plot(container, feature.plot_data, figsize=(8, 5), fontsize=6)
