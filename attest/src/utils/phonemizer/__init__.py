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

from .phonemizer import Phonemizer
from .espeak_phonemizer import EspeakPhonemizer, get_espeak_phonemizer
from .openphonemizer import OpenPhonemizer, get_openphonemizer

from attest.src.settings import get_settings
from attest.src.utils.logger import get_logger


def get_phonemizer() -> Phonemizer:
    settings = get_settings()
    match settings.PHONEMIZATION_METHOD:
        case "espeak_phonemizer":
            return get_espeak_phonemizer()
        case "openphonemizer":
            return get_openphonemizer()
    raise NotImplementedError(f'Phonemization method "{settings.PHONEMIZATION_METHOD}" not implemented')
