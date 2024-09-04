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

import argparse
import json
import os
import sys

from streamlit.web import cli as stcli

sys.path.append(os.path.abspath("attest/third_party/"))
from attest.src import evaluate, compare, multiple_compare  # noqa: E402


def ui():
    sys.argv = ["streamlit", "run", "attest/ui/run.py"]
    sys.exit(stcli.main())


def main():
    parser = argparse.ArgumentParser(description="ATTEST Application")
    subparsers = parser.add_subparsers()

    # Sub-parser for UI
    parser_ui = subparsers.add_parser("ui", help="Run the UI using Streamlit")
    parser_ui.set_defaults(func=ui)

    # Sub-parser for evaluate
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a project with specified features")
    parser_evaluate.add_argument("--project", required=True, help="Name of the project")
    parser_evaluate.add_argument("--features", nargs="+", required=True, help="List of features to evaluate")
    parser_evaluate.add_argument("--output", help="File path to save the result as JSON")
    parser_evaluate.set_defaults(func=evaluate)

    # Sub-parser for compare
    parser_compare = subparsers.add_parser("compare", help="Run the compare method")
    parser_compare.add_argument("--project1", required=True, help="Name of the first project")
    parser_compare.add_argument("--project2", required=True, help="Name of the second project")
    parser_compare.add_argument("--features", nargs="+", required=True, help="List of features to compare")
    parser_compare.add_argument("--output", help="File path to save the result as JSON")
    parser_compare.set_defaults(func=compare)

    # Sub-parser for multiple_compare
    parser_multiple_compare = subparsers.add_parser("multiple_compare", help="Run the multiple compare method")
    parser_multiple_compare.add_argument("--projects", nargs="+", required=True, help="List of projects to compare")
    parser_multiple_compare.add_argument("--features", nargs="+", required=True, help="List of features to compare")
    parser_multiple_compare.add_argument("--output", help="File path to save the result as JSON")
    parser_multiple_compare.set_defaults(func=multiple_compare)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args_dict = vars(args)
        output_file = args_dict.pop("output", None)
        args_dict = {key: value for key, value in args_dict.items() if key != "func"}
        result = args.func(**args_dict)
        result_dict = result.to_dict()
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result_dict, f, indent=4)
        else:
            result_json = json.dumps(result_dict, indent=4)
            print(result_json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
