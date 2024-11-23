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

PROJECT_TITLE = "ATTEST"

HOME_TAB = "Home"
SETTINGS_TAB = "Settings"

EXPORT_LABEL = "**Export**"
EXPORT_BUTTON_LABEL = "Download Metrics as Json"

METHOD_LABEL = "**Method**"
EVALUATE_METHOD = "Evaluate"
COMPARE_METHOD = "Compare"
COMPARE_MULTIPLE_METHOD = "Multiple comparison"

GROUP_AND_PROJECT_LABEL = "**Group and Projects**"
GROUP_LABEL = "Group"
EMPTY_GROUP_LABEL = "None"
PROJECT_LABEL = "Project"
PROJECT_X_LABEL = lambda x: f"Project {x}"  # noqa: E731
NUM_PROJECTS_FOR_MULTIPLE_COMPARE_LABEL = "Number of projects for multiple comparison"

FEATURE_CONFIGURATION_LABEL = "**Feature configuration**"
PHONEMIZATION_METHOD_LABEL = "Phonemization method. "
PITCH_EXTRACT_METHOD_LABEL = "Pitch extraction method. "
'This method is applied to the metrics "VDE/GPE/FFE/logF0 RMSE" and attributes "Pitch mean/std/plot"'
TEXT_NORM_METHOD_LABEL = (
    'Text normalization method. This method is applied to the metrics "Character distance" and "Phoneme distance"'
)

VIEW_CONFIGURATION_LABEL = "**View configuration**"
DISPLAY_ANALYSIS_LABEL = 'Display "Analysis" section'
DISPLAY_SUMMARY_LABEL = 'Display "Summary" section'
ANALYSIS_NUM_COLUMNS_LABEL = "Number of columns with box plots"
SUMMARY_SUBSET_INDEX_LABEL = "Index to select subset after splitting audio name"
DISPLAY_DETAILED_LABEL = 'Display "Detailed" section'
DETAILED_NUM_ITEMS_LABEL = "Items per page"
SHOW_FIRST_COLUMN_LABEL = "Show first column in the Overall table (it is often related to the reference project)"
TRANSPOSE_OVERALL_TABLE_LABEL = "Transpose Overall table (when enabled, projects are columns)"
DISPLAY_COMPARISON_PLOTS_LABEL = "Display graphs with metrics"
TRANSPOSE_COMPARISON_PLOTS_LABEL = "Change axis for graphs with metrics (when enabled, bars are horizontal)"

FEATURES_LABEL = "Features"
SORT_LABEL = "Sort by Metric"
SORT_METHOD_LABEL = "Select Method for Sorting"
SORT_BY_DIFFERENCE_LABEL = "By Absolute Difference"
FILTER_LABEL = "Filter by Metric"
FILTER_MIN_VALUE_LABEL = "Min. Value"
FILTER_MAX_VALUE_LABEL = "Max. Value"
SORT_AND_FILTER_NONE_LABEL = "None"
NAVIGATION_LABEL = "Navigation Controls"
NAVIGATION_PAGE_LABEL = "Page"
NAVIGATION_TOTAL_PAGES_TEXT = lambda x: f"Total Pages: {x}"  # noqa: E731

OVERALL_LABEL = "Overall"
DETAILED_LABEL = "Detailed"
ANALYSIS_LABEL = "Analysis"
ANALYSIS_FIRST_FEATURE_LABEL = "Select first feature"
ANALYSIS_SECOND_FEATURE_LABEL = "Select second feature"
SUMMARY_LABEL = "Summary"
COMPARISON_PLOTS_LABEL = "Graphs with metrics"
COMPARISON_PLOTS_LABEL_X = "Project"
COMPARISON_PLOTS_LABEL_Y = "Values"

ERROR_LABEL = "ERROR"
WARNING_LABEL = "WARNING"
NOT_FOUND_MESSAGE = "Not Found"
FILE_NOT_FOUND = (
    lambda e: f"File not found: {e}. Please make sure the file exists and the path is correct."
)  # noqa: E731
