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

import time
from attest.src.utils.logger import get_logger


class PerformanceTracker:
    def __init__(self, name: str = "", start: bool = False):
        self.name = name
        self.logger = get_logger()
        self.start_time = None
        if start:
            self.start()

    def start(self):
        self.start_time = time.monotonic()
        self.logger.info(f"{self.name} has started.")

    def end(self):
        now = time.monotonic()
        self.logger.info(f"{self.name} has finished. Execution time: {now - self.start_time:.3f} seconds.")
