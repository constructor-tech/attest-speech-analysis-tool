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
import time
import whisper
from whisper.tokenizer import TO_LANGUAGE_CODE

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger


_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = Whisper()

    return _whisper


settings = get_settings()


class Whisper:

    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "whisper", settings.WHISPER_MODEL_NAME)

    def get_model(self):
        if self.model is None:
            start_time = time.time()
            self.logger.info(f"Loading Whisper {settings.WHISPER_MODEL_NAME} model...")

            self.model = whisper.load_model(
                settings.WHISPER_MODEL_NAME,
                device=settings.DEVICE,
                download_root=self.model_cache_dir,
            )

            elapsed_time = time.time() - start_time
            self.logger.info(f"Loaded Whisper {settings.WHISPER_MODEL_NAME} model in {elapsed_time:.2f} seconds")
        return self.model

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/whisper/transcriptions.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def transcribe_project(self, project: Project):
        transcriptions = []
        results = self.infer_project(project)
        for entry in results:
            transcriptions.append(entry["text"])
        return transcriptions

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/whisper/word_alignments.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def align_project(self, project: Project):
        alignments = []
        results = self.infer_project(project)
        for entry in results:
            ali = []
            for segment_entry in entry["segments"]:
                for word_entry in segment_entry["words"]:
                    ali.append(
                        (
                            word_entry["word"],
                            word_entry["start"],
                            word_entry["end"],
                        )
                    )
            alignments.append(ali)
        return alignments

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/whisper/result.json",
        method="json",
        validator=validate_matching_to_project_size,
    )
    def infer_project(self, project: Project):
        self.logger.info("Transcribing speech using Whisper model...")
        results = []

        self.get_model()  # init model

        for audio_file in project.audio_files:

            result = self.model.transcribe(
                audio_file,
                word_timestamps=True,
                fp16=settings.WHISPER_USE_FP16,
                language=TO_LANGUAGE_CODE[settings.WHISPER_LANGUAGE.lower()],
            )
            results.append(result)

        self.logger.info("Transcribing is done!")
        return results
