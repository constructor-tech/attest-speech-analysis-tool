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

import os
import streamlit as st

import attest.ui.constants as vc

from streamlit_option_menu import option_menu

from attest.ui.components.comparison_view import ComparisonView
from attest.ui.components.configuration_view import ConfigurationView
from attest.ui.components.evaluation_view import EvaluationView
from attest.ui.components.multi_comparison_view import MultiComparisonView
from attest.ui.components.sidebar_view import SidebarView
from attest.ui.settings import get_settings
from attest.ui.utils.view_utils import handle_tab_change


class View:
    def __init__(self):
        self.settings = get_settings()
        self.configuration_view = ConfigurationView(self.settings)
        self.sidebar_view = SidebarView(self.settings)
        self.evaluation_view = EvaluationView(self.settings)
        self.comparison_view = ComparisonView(self.settings)
        self.multi_comparison_view = MultiComparisonView(self.settings)

        self.projects = []

        favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
        st.set_page_config(page_title=vc.PROJECT_TITLE, page_icon=favicon_path, layout="wide")

        self.setup_menu()
        self.setup_sidebar()

    def setup_menu(self):
        option_menu(
            None,
            [vc.HOME_TAB, vc.CONFIGURATION_TAB],
            icons=["house", "gear"],
            menu_icon="cast",
            default_index=1,
            orientation="horizontal",
            key="tab",
            on_change=lambda _: handle_tab_change(st.session_state),
        )

    def setup_sidebar(self):
        self.sidebar_view.setup_sidebar()

    def display_settings(self):
        self.configuration_view.display_settings()

    def display_error(self, msg):
        st.error(f"⚠️ {vc.ERROR_LABEL}: {msg}")

    def display_evaluation_result(self):
        self.sidebar_view.display_sidebar()
        self.evaluation_view.display_evaluation_result(current_page=self.sidebar_view.current_page)

    def display_comparison_result(self):
        self.sidebar_view.display_sidebar()
        self.comparison_view.display_comparison_result(current_page=self.sidebar_view.current_page)

    def display_multi_comparison_result(self):
        self.multi_comparison_view.display_multi_comparison_result()
