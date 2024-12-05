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

import matplotlib.pyplot as plt
import streamlit as st

import attest.ui.constants as vc
from attest.ui.model import UIMultipleComparisonResult
from attest.ui.settings import Settings
from attest.ui.utils.view_utils import (
    convert_to_dataframe,
    convert_to_table,
    dataframe_to_latex,
    dataframe_to_markdown,
    get_project_name,
)


class MultiComparisonView:

    def __init__(self, settings: Settings):
        self.settings = settings

    def display_multi_comparison_result(self):
        result: UIMultipleComparisonResult = st.session_state.parsed_result

        project_ids = result.projects
        feature_ids = result.overall_metrics.keys()
        project_names = [get_project_name(x) for x in result.projects]
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
