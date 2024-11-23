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


def validate_matching_to_project_size(cached_data, cls, project, *args, **kwargs):
    return len(cached_data) == len(project)


def validate_feature_from_cache(feature_result, feature_id, project, cache_filename):
    feature_result_uids = [x.uid for x in feature_result.detailed]
    cache_is_valid = len(feature_result_uids) == len(project.uids) and all(
        uid in feature_result_uids for uid in project.uids
    )
    return cache_is_valid


def validate_reference_aware_feature_from_cache(feature_result, feature_id, hyp_project, ref_project, cache_filename):
    feature_result_uids = [x.uid for x in feature_result.detailed]
    cache_is_valid = len(feature_result_uids) == len(hyp_project.uids) and all(
        uid in feature_result_uids for uid in hyp_project.uids
    )
    return cache_is_valid
