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

import logging
import os
from attest.ui.settings import get_settings


_logger = None


def get_logger():
    global _logger
    if _logger is None:
        settings = get_settings()

        logging.basicConfig(format="%(asctime)s %(levelname)s %(module)s: %(message)s")
        logging_level = logging.getLevelName(settings.LOGGING_LEVEL)
        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging_level)

    return _logger


def check_if_group(path):
    num_projects = len(get_list_of_projects(path))
    return num_projects > 0


def check_if_project(path):
    file_path = f"{path}/meta/filelist.txt"
    return os.path.exists(file_path)


def resolve_group_path(data_dir, group=None):
    return f"{data_dir}/{group}" if group else data_dir


def get_list_of_groups(path):
    return [x for x in os.listdir(path) if check_if_group(f"{path}/{x}")]


def get_list_of_projects(path):
    return [x for x in os.listdir(path) if check_if_project(f"{path}/{x}")]


def get_list_of_pitch_extract_methods():
    methods = ["parselmouth", "pyworld", "torchcrepe-tiny", "torchcrepe-full"]
    return methods


def get_list_of_text_norm_methods():
    methods = ["None"]
    try:
        from nemo_text_processing.text_normalization import Normalizer  # noqa: F401

        methods.append("Nemo")
    except ImportError:
        pass
    return methods


def get_list_of_phonemization_methods():
    methods = ["openphonemizer", "espeak_phonemizer"]
    return methods
