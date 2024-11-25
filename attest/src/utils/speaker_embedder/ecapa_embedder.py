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
from speechbrain.pretrained import EncoderClassifier

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.audio_utils import load_audio_tensor
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.performance_tracker import PerformanceTracker
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
        self.model = None
        self.model_name = "spkrec-ecapa-voxceleb"
        self.model_cache_dir = os.path.join(settings.MODELS_DIR, "speaker_embeddings/spkrec-ecapa-voxceleb")
        self.sampling_rate = 16000
        self.device = settings.DEVICE

    def load_model(self):
        if self.model is None:
            tracker = PerformanceTracker(name="Loading spkrec-ecapa-voxceleb model", start=True)
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.model_cache_dir,
                run_opts={"device": self.device},
            )
            tracker.end()

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/speaker_embeddings/ecapa_embeddings.pt",
        method="torch",
        validator=validate_matching_to_project_size,
    )
    def extract_embeddings_for_project(self, project: Project):
        self.load_model()

        tracker = PerformanceTracker(name="Extracting speaker embeddings", start=True)
        embeddings = []
        for audio_path in project.audio_files:
            audio_tensor, _ = load_audio_tensor(audio_path, target_sr=self.sampling_rate, target_channels=1, device=self.device)
            embedding = self.model.encode_batch(audio_tensor)[0][0].cpu()
            embeddings.append(embedding)
        tracker.end()

        return embeddings

    def compute_similarity(self, x, y):
        if x.device != y.device:
            x = x.cpu()
            y = y.cpu()
        return x @ y / (x.norm() * y.norm())
