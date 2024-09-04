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

from .text_normalizer import TextNormalizer, DummyTextNormalizer
from .nemo_normalizer import NemoTextNormalizer, get_nemo_text_normalizer

from attest.src.settings import get_settings
from attest.src.utils.logger import get_logger


def get_text_normalizer():
    logger = get_logger()
    settings = get_settings()
    match settings.TEXT_NORM_METHOD:
        case "None":
            return DummyTextNormalizer()
        case "Nemo":
            try:
                return get_nemo_text_normalizer()
            except ImportError:
                logger.warning(
                    "nemo_text_processing not found. "
                    "Please install NeMo-text-processing: https://github.com/NVIDIA/NeMo-text-processing"
                )
                return DummyTextNormalizer()
    logger.warning(f'Text normalization "{settings.TEXT_NORM_METHOD}" not implemented')
    return DummyTextNormalizer()
