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


def edit_distance(seq1, seq2) -> int:
    n, m = len(seq1), len(seq2)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    min(d[i][j - 1] + 1, d[i - 1][j] + 1),  # insert  # delete
                    d[i - 1][j - 1] + 1,  # replace
                )
    return d[n][m]


def edit_distance_many(seqs1, seqs2):
    return [edit_distance(seq1, seq2) for seq1, seq2 in zip(seqs1, seqs2)]


def format_text_for_edit_distance(text, remove_tags=True):
    # lowercase
    text = text.lower()

    # expand abbreviations
    abbreviations = [
        ("mr", "mister"),
        ("mrs", "missus"),
        ("dr", "doctor"),
        ("etc", "et cetera"),
    ]
    text = " %s " % text
    for short, long in abbreviations:
        text = text.replace(" %s " % short, " %s " % long)
    text = text.strip()

    # replace bad symbols
    replacements = {
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
        "«": '"',
        "»": '"',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # remove tags
    if remove_tags:
        text_no_tags = []
        in_tag = False
        for c in text:
            if c == "<":
                in_tag = True
            elif c == ">":
                in_tag = False
            elif not in_tag:
                text_no_tags.append(c)
        text = "".join(text_no_tags)

    # remove punctuation and spaces:
    for c in "!?:;.,\"-'()[] ":
        text = text.replace(c, "")

    return text
