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
import numpy as np
import pandas as pd
import streamlit as st


def display_plot(st_object, plot, figsize, fontsize):
    values = [float(value) if value != "NaN" else float("nan") for value in plot["values"]]

    plt.figure(figsize=figsize)
    plt.plot(values)
    plt.title(plot["title"])
    plt.xlabel("Index")
    plt.ylabel("Value")

    if "plot_labels" in plot:
        for label, x, y in plot["plot_labels"]:
            plt.text(x, y, label, fontsize=fontsize, ha="center")

    if "horizontal_segments" in plot:
        for y, x0, x1 in plot["horizontal_segments"]:
            plt.hlines(y=y, xmin=x0, xmax=x1, color="r")

    if "vertical_lines" in plot:
        for x in plot["vertical_lines"]:
            plt.axvline(x=x, color="r", linestyle="--")

    st_object.pyplot(plt)


def display_boxplot(st_object, data, labels, figsize, title):
    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=labels, vert=True, showmeans=True, meanline=True)
    plt.title(title)
    plt.ylabel("Values")
    st_object.pyplot(plt)


def display_correlation_plot(feature1_id, feature2_id, feature1_values, feature2_values):
    feature1_values = np.array(feature1_values)
    feature2_values = np.array(feature2_values)

    plt.figure(figsize=(12, 12))
    plt.scatter(feature1_values, feature2_values, alpha=0.5)
    plt.xlabel(feature1_id)
    plt.ylabel(feature2_id)
    plt.title(f"Correlation between {feature1_id} and {feature2_id}")

    corr_coef = np.corrcoef(feature1_values, feature2_values)[0, 1]
    a, b = np.polyfit(feature1_values, feature2_values, 1)
    sorted_indices = np.argsort(feature1_values)
    sorted_feature1_values = feature1_values[sorted_indices]
    sorted_feature2_values = a * sorted_feature1_values + b
    plt.plot(sorted_feature1_values, sorted_feature2_values, color="red")

    st.pyplot(plt)
    st.write(f"Correlation Coefficient: {corr_coef:.3f}")


def convert_to_table(array2d):
    table_html = "<table>"
    for row in array2d:
        table_html += "<tr>"
        for cell in row:
            if isinstance(cell, float):
                cell = "%.3f" % cell
            table_html += f"<td>{cell}</td>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html


def convert_to_dataframe(array2d):
    headers = array2d[0]
    rows = array2d[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df


def dataframe_to_markdown(df):
    return df.to_markdown(index=False)


def dataframe_to_latex(df):
    return df.to_latex(index=False)
