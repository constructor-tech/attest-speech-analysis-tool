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

import functools
import json
import os
import pickle
import re
import torch

from attest.src.utils.logger import get_logger


class CacheHandler:

    def __init__(self, cache_path_template, method, validator=None):
        self.cache_path_template = cache_path_template
        self.method = method
        self.validator = validator
        self.logger = get_logger()

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_file = self._parse_cache_path(self.cache_path_template, *args, **kwargs)
            if os.path.exists(cache_file):
                self.logger.info(f'Found cached file: "{cache_file}"')
                cached_data = self._load_cache(cache_file)
                cache_is_valid = True
                if self.validator:
                    cache_is_valid &= self.validator(cached_data, *args, **kwargs)
                if cache_is_valid:
                    self.logger.info("Used data from cache!")
                    return cached_data
                else:
                    self.logger.info("Cache is not valid, recomputing.")

            result = func(*args, **kwargs)
            self._save_cache(result, cache_file)
            self.logger.info(f'Cached data to "{cache_file}"')
            return result

        return wrapper

    def _load_cache(self, cache_file):
        if self.method == "torch":
            return torch.load(cache_file)
        elif self.method == "pickle":
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        elif self.method == "json":
            with open(cache_file, "r") as f:
                return json.load(f)
        elif self.method == "txt":
            with open(cache_file, "r") as f:
                return [x.strip() for x in f]
        else:
            raise ValueError("Unsupported caching method")

    def _save_cache(self, data, cache_file):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if self.method == "torch":
            torch.save(data, cache_file)
        elif self.method == "pickle":
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        elif self.method == "json":
            with open(cache_file, "w") as f:
                json.dump(data, f)
        elif self.method == "txt":
            with open(cache_file, "w") as f:
                for t in data:
                    f.write(f"{t}\n")
        else:
            raise ValueError("Unsupported caching method")

    def _parse_cache_path(self, template, *args, **kwargs):
        # TODO @od 20.02.2024: Take kwargs into account
        #                       Only arguments from args could now be used for cache path
        def replacer(match):
            index, attribute = match.groups()
            index = int(index)
            if attribute:
                return str(getattr(args[index], attribute))
            return str(args[index])

        pattern = r"\${(\d+)(?:\.([a-zA-Z_][a-zA-Z0-9_]*))?}"
        return re.sub(pattern, replacer, template)
