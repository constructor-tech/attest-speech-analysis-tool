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
from attest.ui.model import UIEvaluationResult
from attest.ui.settings import Settings
from attest.ui.utils.view_utils import (
    display_detailed_fetuare,
    display_boxplot,
    display_correlation_plot,
    get_project_name,
)


class EvaluationView:

    def __init__(self, settings: Settings):
        self.settings = settings

    def display_evaluation_result(self, current_page: int):
        result: UIEvaluationResult = st.session_state.parsed_result

        metric_keys = [key for key in self.settings.FEATURES if key in result.overall_metrics]
        detailed_list = list(result.detailed_features.values())

        # Main field
        project_name = get_project_name(result.project)
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
            start_index = (current_page - 1) * self.settings.ITEMS_PER_PAGE
            end_index = min(start_index + self.settings.ITEMS_PER_PAGE, len(detailed_list))

            for entry in detailed_list[start_index:end_index]:
                for feature_id in self.settings.FEATURES:
                    if feature_id not in entry:
                        continue
                    feature = entry[feature_id]
                    display_detailed_fetuare(
                        feature_label=self.settings.FEATURE_ID_TO_LABEL[feature_id],
                        feature=feature,
                        is_metric=feature_id in metric_keys,
                        container=st,
                    )
                st.write("")
                st.write("")
