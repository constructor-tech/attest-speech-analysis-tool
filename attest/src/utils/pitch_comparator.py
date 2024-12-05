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

# part of the code is borrowed from
# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_f0.py

import librosa
import numpy as np
import pysptk

from fastdtw import fastdtw
from scipy import spatial

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.performance_tracker import PerformanceTracker
from attest.src.utils.pitch_extractor import get_pitch_extractor


_pitch_comparator = None
settings = get_settings()


class PitchComparator:

    def compare_for_projects(self, hyp_project: Project, ref_project: Project, key: str):
        result = self._compare_for_projects(hyp_project, ref_project, settings.PITCH_EXTRACT_METHOD)
        return [x[key] for x in result]

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/pitch_comparator/{{3}}/${{2.name}}.pickle",
        method="pickle",
        validator=validate_matching_to_project_size,
    )
    def _compare_for_projects(self, hyp_project: Project, ref_project: Project, _pitch_extract_method: str):
        pitch_extractor = get_pitch_extractor()
        pitch_values_hyp = pitch_extractor.compute_pitch_values_for_project(hyp_project)
        pitch_values_ref = pitch_extractor.compute_pitch_values_for_project(ref_project)

        tracker = PerformanceTracker(name="Comparing pitch values", start=True)
        result = []
        for audio_ref, audio_syn, pitch_ref, pitch_syn in zip(
            ref_project.audio_files,
            hyp_project.audio_files,
            pitch_values_ref,
            pitch_values_hyp,
        ):
            result.append(
                self.compute_pitch_metrics(
                    audio_ref=audio_ref,
                    audio_syn=audio_syn,
                    pitch_ref=pitch_ref,
                    pitch_syn=pitch_syn,
                )
            )
        tracker.end()

        return result

    def compute_pitch_metrics(
        self,
        audio_ref: str,
        audio_syn: str,
        pitch_ref: np.array,
        pitch_syn: np.array,
        delta_f0e=0.2,
    ):
        wav_ref, sr = librosa.load(audio_ref)
        wav_syn, sr = librosa.load(audio_syn)
        wav_ref_trimmed, _ = librosa.effects.trim(wav_ref, top_db=50)
        wav_syn_trimmed, _ = librosa.effects.trim(wav_syn, top_db=50)

        twf, shapes = self.compute_path_sptk(wav_syn_trimmed, wav_ref_trimmed, sr)

        pitch_ref = np.nan_to_num(pitch_ref)
        pitch_syn = np.nan_to_num(pitch_syn)
        pitch_ref_dtw = pitch_ref[twf[1]]
        pitch_syn_dtw = pitch_syn[twf[0]]

        nonzero_idxs = np.where((pitch_ref_dtw != 0) & (pitch_syn_dtw != 0))[0]
        pitch_ref_dtw_voiced = np.log(pitch_ref_dtw[nonzero_idxs])
        pitch_syn_dtw_voiced = np.log(pitch_syn_dtw[nonzero_idxs])

        mask_ref = pitch_ref_dtw > 0
        mask_syn = pitch_syn_dtw > 0

        N_uv_vu = sum(mask_ref ^ mask_syn)
        N = pitch_syn_dtw.shape[0]
        N_f0e = sum(
            abs(p_syn / p_ref - 1) > delta_f0e
            for p_ref, p_syn in zip(pitch_ref_dtw, pitch_syn_dtw)
            if p_ref > 0 and p_syn > 0
        )
        N_vv = sum(mask_ref & mask_syn)

        VDE = N_uv_vu / N * 100
        GPE = N_f0e / N_vv * 100
        FFE = GPE * (N_vv / N) + VDE
        log_F0_RMSE = np.sqrt(np.mean((pitch_syn_dtw_voiced - pitch_ref_dtw_voiced) ** 2))

        return {
            "VDE": VDE,
            "GPE": GPE,
            "FFE": FFE,
            "log_F0_RMSE": log_F0_RMSE,
        }

    def compute_path_sptk(self, wav_gen, wav_gt, sr):
        # extract ground truth and converted features
        gen_mcep = self.sptk_extract(
            x=wav_gen,
            fs=sr,
            # n_fft=args.n_fft,
            # n_shift=args.n_shift,
            # mcep_dim=args.mcep_dim,
            # mcep_alpha=args.mcep_alpha,
        )
        gt_mcep = self.sptk_extract(
            x=wav_gt,
            fs=sr,
            # n_fft=args.n_fft,
            # n_shift=args.n_shift,
            # mcep_dim=args.mcep_dim,
            # mcep_alpha=args.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T

        return twf, (gen_mcep.shape[0], gt_mcep.shape[0])

    def sptk_extract(
        self,
        x: np.ndarray,
        fs: int,
        n_fft: int = 512,
        n_shift: int = 256,
        mcep_dim: int = 25,
        mcep_alpha: float = 0.41,
        is_padding: bool = False,
    ) -> np.ndarray:
        """Extract SPTK-based mel-cepstrum.
        Args:
            x (ndarray): 1D waveform array.
            fs (int): Sampling rate
            n_fft (int): FFT length in point (default=512).
            n_shift (int): Shift length in point (default=256).
            mcep_dim (int): Dimension of mel-cepstrum (default=25).
            mcep_alpha (float): All pass filter coefficient (default=0.41).
            is_padding (bool): Whether to pad the end of signal (default=False).
        Returns:
            ndarray: Mel-cepstrum with the size (N, n_fft).
        """
        # perform padding
        if is_padding:
            n_pad = n_fft - (len(x) - n_fft) % n_shift
            x = np.pad(x, (0, n_pad), "reflect")

        # get number of frames
        n_frame = (len(x) - n_fft) // n_shift + 1

        # get window function
        win = pysptk.sptk.hamming(n_fft)

        # check mcep and alpha
        if mcep_dim is None or mcep_alpha is None:
            mcep_dim, mcep_alpha = self._get_best_mcep_params(fs)

        # calculate spectrogram
        mcep = [
            pysptk.mcep(
                x[n_shift * i : n_shift * i + n_fft] * win,
                mcep_dim,
                mcep_alpha,
                eps=1e-6,
                etype=1,
            )
            for i in range(n_frame)
        ]

        return np.stack(mcep)

    def _get_best_mcep_params(self, fs: int):
        if fs == 16000:
            return 23, 0.42
        elif fs == 22050:
            return 34, 0.45
        elif fs == 24000:
            return 34, 0.46
        elif fs == 44100:
            return 39, 0.53
        elif fs == 48000:
            return 39, 0.55
        else:
            raise ValueError(f"Not found the setting for {fs}.")


def get_pitch_comparator() -> PitchComparator:
    global _pitch_comparator
    if _pitch_comparator is None:
        _pitch_comparator = PitchComparator()

    return _pitch_comparator
