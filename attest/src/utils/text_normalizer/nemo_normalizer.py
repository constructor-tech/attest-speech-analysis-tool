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
from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from attest.src.utils.performance_tracker import PerformanceTracker


settings = get_settings()
_text_normalizer = None


class NemoTextNormalizer:

    def __init__(self):
        self.logger = get_logger()
        self.models = {}
        self.runtime_cache = {}
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "text_norm/nemo")
        self.available_langs = [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "ar",  # Arabic
            "sv",  # Swedish
            "zh",  # Chinese
            "hu",  # Hungarian
            "it",  # Italian
        ]
        # list of supported languages for text normalization from NeMo documentation:
        #   https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html#language-support-matrix

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/text_norm/nemo/normalized_texts.txt",
        method="txt",
        validator=validate_matching_to_project_size,
    )
    def normalize_project(self, project: Project):
        tracker = PerformanceTracker(name="Normalizing texts", start=True)
        normalized_texts = [self.normalize(text) for text in project.texts]
        tracker.end()

        return normalized_texts

    def get_model(self, lang):
        if lang not in self.available_langs:
            self.logger.info('Text normalization for language "{lang}" is not supported.')
            return None

        if lang not in self.models:
            from nemo_text_processing.text_normalization.normalize import Normalizer

            tracker = PerformanceTracker(name=f"Loading Text Normalization model for language '{lang}'", start=True)
            self.models[lang] = Normalizer(input_case="cased", lang=lang, cache_dir=self.model_cache_dir)
            tracker.end()

        return self.models[lang]

    def normalize(self, text, lang="en"):
        if text not in self.runtime_cache:
            text, has_bad_symbol = self._replace_bad_symbols(text)
            text_normalized = self.get_model(lang).normalize(text, punct_post_process=True)
            self.runtime_cache[text] = text_normalized
        return self.runtime_cache[text]

    def _replace_bad_symbols(self, text):
        replacements = {
            "’": "'",
            "‘": "'",
            "—": "-",
            "–": "-",
            "«": '"',
            "»": '"',
        }
        cleaned_text = text
        for x, y in replacements.items():
            cleaned_text = cleaned_text.replace(x, y)
        return cleaned_text, text == cleaned_text


def get_nemo_text_normalizer() -> NemoTextNormalizer:
    global _text_normalizer
    if _text_normalizer is None:
        _text_normalizer = NemoTextNormalizer()

    return _text_normalizer
