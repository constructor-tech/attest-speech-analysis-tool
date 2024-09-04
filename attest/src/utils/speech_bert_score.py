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

import torch


def bert_score(v_generated, v_reference):
    """
    Source:
        https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics/blob/master/discrete_speech_metrics/speechbertscore.py#L15
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """
    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (
        torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0)
    )

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean().item()
    recall = torch.max(sim_matrix, dim=0)[0].mean().item()

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score
