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
from attest.ui.model import UIComparisonResult
from attest.ui.settings import Settings
from attest.ui.utils.view_utils import (
    convert_to_table,
    display_detailed_fetuare,
    get_project_name,
)


class ComparisonView:

    def __init__(self, settings: Settings):
        self.settings = settings

    def display_comparison_result(self, current_page: int):
        result: UIComparisonResult = st.session_state.parsed_result

        metric_keys = [key for key in self.settings.FEATURES if key in result.overall_metrics]
        project1_name = get_project_name(result.project1)
        project2_name = get_project_name(result.project2)
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
            start_index = (current_page - 1) * self.settings.ITEMS_PER_PAGE
            end_index = min(start_index + self.settings.ITEMS_PER_PAGE, len(detailed_list))

            for entry in detailed_list[start_index:end_index]:
                cols = st.columns(2)
                for j in range(2):
                    for feature_id in self.settings.FEATURES:
                        if feature_id not in entry:
                            continue
                        feature = entry[feature_id][j]
                        display_detailed_fetuare(
                            feature_label=self.settings.FEATURE_ID_TO_LABEL[feature_id],
                            feature=feature,
                            is_metric=feature_id in metric_keys,
                            container=cols[j],
                        )
                st.write("")
                st.write("")
