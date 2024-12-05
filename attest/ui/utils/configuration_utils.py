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


def check_if_group(path):
    num_projects = len(get_list_of_projects(path))
    return num_projects > 0


def check_if_project(path):
    file_path = f"{path}/meta/filelist.txt"
    return os.path.exists(file_path)


def resolve_group_path(data_dir, group=None):
    return f"{data_dir}/{group}" if group else data_dir


def get_list_of_groups(path):
    return [x for x in os.listdir(path) if check_if_group(f"{path}/{x}")]


def get_list_of_projects(path):
    return [x for x in os.listdir(path) if check_if_project(f"{path}/{x}")]


def get_list_of_pitch_extract_methods():
    methods = ["parselmouth", "pyworld", "torchcrepe-tiny", "torchcrepe-full"]
    return methods


def get_list_of_text_norm_methods():
    methods = ["None"]
    try:
        from nemo_text_processing.text_normalization import Normalizer  # noqa: F401

        methods.append("Nemo")
    except ImportError:
        pass
    return methods


def get_list_of_phonemization_methods():
    methods = ["openphonemizer", "espeak_phonemizer"]
    return methods


def get_list_of_languages_whisper():
    from whisper.tokenizer import TO_LANGUAGE_CODE

    languages = list(TO_LANGUAGE_CODE.keys())
    languages = [x[0].upper() + x[1:] for x in languages]
    return languages


def get_list_of_languages_espeak():
    return [
        "English",
        "Afrikaans",
        "Amharic",
        "Aragonese",
        "Arabic",
        "Assamese",
        "Azerbaijani",
        "Bashkir",
        "Bulgarian",
        "Bengali",
        "Bishnupriya_Manipuri",
        "Bosnian",
        "Catalan",
        "Chinese_(Mandarin)",
        "Czech",
        "Welsh",
        "Danish",
        "German",
        "Greek",
        "English_(Caribbean)",
        "English_(Great_Britain)",
        "English_(Scotland)",
        "English_(Lancaster)",
        "English_(West_Midlands)",
        "English_(Received_Pronunciation)",
        "English_(America)",
        "Esperanto",
        "Spanish_(Spain)",
        "Spanish_(Latin_America)",
        "Estonian",
        "Basque",
        "Persian",
        "Persian_(Pinglish)",
        "Finnish",
        "French_(Belgium)",
        "French_(Switzerland)",
        "French_(France)",
        "Gaelic_(Irish)",
        "Gaelic_(Scottish)",
        "Guarani",
        "Greek_(Ancient)",
        "Gujarati",
        "Hakka_Chinese",
        "Hindi",
        "Croatian",
        "Haitian_Creole",
        "Hungarian",
        "Armenian_(East_Armenia)",
        "Armenian_(West_Armenia)",
        "Interlingua",
        "Indonesian",
        "Icelandic",
        "Italian",
        "Japanese",
        "Lojban",
        "Georgian",
        "Kazakh",
        "Greenlandic",
        "Kannada",
        "Korean",
        "Konkani",
        "Kurdish",
        "Kyrgyz",
        "Latin",
        "Lingua_Franca_Nova",
        "Lithuanian",
        "Latvian",
        "Māori",
        "Macedonian",
        "Malayalam",
        "Marathi",
        "Malay",
        "Maltese",
        "Myanmar_(Burmese)",
        "Norwegian_Bokmål",
        "Nahuatl_(Classical)",
        "Nepali",
        "Dutch",
        "Oromo",
        "Oriya",
        "Punjabi",
        "Papiamento",
        "Polish",
        "Portuguese_(Portugal)",
        "Portuguese_(Brazil)",
        "Pyash",
        "K'iche'",
        "Romanian",
        "Russian",
        "Russian_(Latvia)",
        "Sindhi",
        "Shan_(Tai_Yai)",
        "Sinhala",
        "Slovak",
        "Slovenian",
        "Albanian",
        "Serbian",
        "Swedish",
        "Swahili",
        "Tamil",
        "Telugu",
        "Setswana",
        "Turkish",
        "Tatar",
        "Urdu",
        "Uzbek",
        "Vietnamese_(Northern)",
        "Vietnamese_(Central)",
        "Vietnamese_(Southern)",
        "Chinese_(Cantonese)",
    ]
