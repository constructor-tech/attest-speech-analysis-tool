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
