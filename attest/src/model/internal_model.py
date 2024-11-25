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
from typing import List

from attest.src.settings import get_settings
from attest.src.utils.logger import get_logger
from attest.src.utils.performance_tracker import PerformanceTracker


# Initialize settings and logger
settings = get_settings()
logger = get_logger()


class Project:
    """Class representing a project with audio files and their corresponding text labels."""

    def __init__(self, name: str, filelist: List[List[str]], audio_dir: str):
        self.name = name
        self.filelist = filelist
        self.audio_dir = audio_dir

    @property
    def uids(self) -> List[str]:
        """List of unique identifiers for each project entry."""
        return [entry[0] for entry in self.filelist]

    @property
    def audio_files(self) -> List[str]:
        """List of paths to audio files."""
        return [os.path.join(self.audio_dir, entry[0]) for entry in self.filelist]

    @property
    def texts(self) -> List[str]:
        """List of textual transcriptions associated with each audio file."""
        return [entry[1] for entry in self.filelist]

    def __len__(self) -> int:
        """Return the number of entries in the project."""
        return len(self.filelist)


def load_project(project_id: str) -> Project:
    """Load a project by its ID.

    Args:
        project_id (str): The identifier for the project to load.

    Returns:
        Project: The loaded project object.

    Raises:
        FileNotFoundError: If the project directory or filelist does not exist.
    """
    project_path = os.path.join(settings.DATA_DIR, project_id)

    tracker = PerformanceTracker(name=f"Loading project '{project_id}' from '{project_path}'", start=True)

    # Ensure project directory exists
    if not os.path.exists(project_path):
        error_msg = f"Project '{project_path}' not found!"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Ensure the filelist exists
    filelist_path = os.path.join(project_path, "meta/filelist.txt")
    if not os.path.exists(filelist_path):
        error_msg = f"Filelist '{filelist_path}' not found!"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Read the filelist
    with open(filelist_path, "r") as file:
        filelist = [line.strip().split("|") for line in file]

    tracker.end()

    # Return the initialized Project object
    return Project(project_id, filelist, audio_dir=os.path.join(project_path, "wavs"))


def validate_project_alignment(project1: Project, project2: Project) -> (Project, Project):
    """Validates that two projects are aligned by ensuring they have the same number of entries and matching keys.

    Args:
        project1 (Project): The first project to be validated.
        project2 (Project): The second project to be validated.

    Returns:
        Tuple[Project, Project]: The original projects if they are aligned correctly.

    Raises:
        AssertionError: If the projects have different number of entries or mismatched keys.
    """

    # Check if both projects have the same number of files
    num_files1 = len(project1.filelist)
    num_files2 = len(project2.filelist)
    assert (
        num_files1 == num_files2
    ), f"Error when aligning projects: different number of entries in projects {num_files1} != {num_files2}"

    # Check if all corresponding entries have matching keys
    for index, (file1, file2) in enumerate(zip(project1.filelist, project2.filelist)):
        assert file1[0] == file2[0], f"Error when aligning projects: key #{index} mismatch {file1[0]} != {file2[0]}"

    return project1, project2
