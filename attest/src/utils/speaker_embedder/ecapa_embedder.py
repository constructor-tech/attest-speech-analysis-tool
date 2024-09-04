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
import torchaudio

from speechbrain.pretrained import EncoderClassifier

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from .speaker_embedder import SpeakerEmbedder


_ecapa_embedder = None


def get_ecapa_embedder():
    global _ecapa_embedder
    if _ecapa_embedder is None:
        _ecapa_embedder = ECAPAEmbedder()

    return _ecapa_embedder


settings = get_settings()


class ECAPAEmbedder(SpeakerEmbedder):

    def __init__(self):
        self.logger = get_logger()
        self.model = None
        self.model_name = "spkrec-ecapa-voxceleb"
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "speaker_embeddings/spkrec-ecapa-voxceleb")
        self.sampling_rate = 16_000
        self.device = settings.DEVICE

    def get_model(self):
        if self.model is None:
            start_time = time.time()
            self.logger.info("Loading spkrec-ecapa-voxceleb model...")

            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.model_cache_dir,
                run_opts={"device": self.device},
            )
            self.logger.info("Loaded spkrec-ecapa-voxceleb model in %.2f seconds" % (time.time() - start_time))
        return self.model

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/speaker_embeddings/ecapa_embeddings.pt",
        method="torch",
        validator=validate_matching_to_project_size,
    )
    def extract_embeddings_for_project(self, project: Project):
        self.logger.info("Extracting speaker embeddings...")
        embeddings = []

        for x in project.audio_files:
            signal, sr = torchaudio.load(x)
            signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=self.sampling_rate)
            embedding = self.get_model().encode_batch(signal)[0][0].cpu()
            embeddings.append(embedding)

        self.logger.info("Extracting speaker embeddings is done!")
        return embeddings

    def compute_similarity(self, x, y):
        if x.device != y.device:
            x = x.cpu()
            y = y.cpu()
        return x @ y / (x.norm() * y.norm())
