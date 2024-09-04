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

from .pitch_extractor import PitchExtractor
from .pyworld_pitch_extractor import get_pyworld_pitch_extractor, PyworldPitchExtractor
from .torchcrepe_pitch_extractor import (
    get_torchcrepe_pitch_extractor,
    CrepePitchExtractor,
)

from attest.src.settings import get_settings
from attest.src.utils.logger import get_logger


def get_pitch_extractor() -> PitchExtractor:
    logger = get_logger()
    settings = get_settings()
    match settings.PITCH_EXTRACT_METHOD:
        case "torchcrepe-tiny":
            return get_torchcrepe_pitch_extractor("tiny")
        case "torchcrepe-full":
            return get_torchcrepe_pitch_extractor("full")
        case "pyworld":
            return get_pyworld_pitch_extractor()
    logger.warning(f'Pitch extraction method "{settings.PITCH_EXTRACT_METHOD}" not implemented')
    return get_torchcrepe_pitch_extractor()
