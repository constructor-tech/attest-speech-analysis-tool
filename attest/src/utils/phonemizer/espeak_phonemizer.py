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

from typing import List

from attest.src.settings import get_settings
from attest.src.model import Project
from attest.src.utils.caching_utils import CacheHandler
from attest.src.utils.caching_validators import validate_matching_to_project_size
from attest.src.utils.logger import get_logger
from phoneme_tokenizer import PhonemeTokenizer  # third_party
from .phonemizer import Phonemizer


_phonemizer = None


def get_espeak_phonemizer():
    global _phonemizer
    if _phonemizer is None:
        _phonemizer = EspeakPhonemizer()

    return _phonemizer


settings = get_settings()


TO_LANGUAGE_CODE = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Aragonese": "an",
    "Arabic": "ar",
    "Assamese": "as",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Bishnupriya_Manipuri": "bpy",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Chinese_(Mandarin)": "cmn",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English_(Caribbean)": "en-029",
    "English_(Great_Britain)": "en-gb",
    "English_(Scotland)": "en-gb-scotland",
    "English_(Lancaster)": "en-gb-x-gbclan",
    "English_(West_Midlands)": "en-gb-x-gbcwmd",
    "English_(Received_Pronunciation)": "en-gb-x-rp",
    "English_(America)": "en-us",
    "Esperanto": "eo",
    "Spanish_(Spain)": "es",
    "Spanish_(Latin_America)": "es-419",
    "Estonian": "et",
    "Basque": "eu",
    "Persian": "fa",
    "Persian_(Pinglish)": "fa-latn",
    "Finnish": "fi",
    "French_(Belgium)": "fr-be",
    "French_(Switzerland)": "fr-ch",
    "French_(France)": "fr-fr",
    "Gaelic_(Irish)": "ga",
    "Gaelic_(Scottish)": "gd",
    "Guarani": "gn",
    "Greek_(Ancient)": "grc",
    "Gujarati": "gu",
    "Hakka_Chinese": "hak",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian_Creole": "ht",
    "Hungarian": "hu",
    "Armenian_(East_Armenia)": "hy",
    "Armenian_(West_Armenia)": "hyw",
    "Interlingua": "ia",
    "Indonesian": "id",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Lojban": "jbo",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Greenlandic": "kl",
    "Kannada": "kn",
    "Korean": "ko",
    "Konkani": "kok",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Latin": "la",
    "Lingua_Franca_Nova": "lfn",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Māori": "mi",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Malay": "ms",
    "Maltese": "mt",
    "Myanmar_(Burmese)": "my",
    "Norwegian_Bokmål": "nb",
    "Nahuatl_(Classical)": "nci",
    "Nepali": "ne",
    "Dutch": "nl",
    "Oromo": "om",
    "Oriya": "or",
    "Punjabi": "pa",
    "Papiamento": "pap",
    "Polish": "pl",
    "Portuguese_(Portugal)": "pt",
    "Portuguese_(Brazil)": "pt-br",
    "Pyash": "py",
    "K'iche'": "quc",
    "Romanian": "ro",
    "Russian": "ru",
    "Russian_(Latvia)": "ru-lv",
    "Sindhi": "sd",
    "Shan_(Tai_Yai)": "shn",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Setswana": "tn",
    "Turkish": "tr",
    "Tatar": "tt",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese_(Northern)": "vi",
    "Vietnamese_(Central)": "vi-vn-x-central",
    "Vietnamese_(Southern)": "vi-vn-x-south",
    "Chinese_(Cantonese)": "yue",
}


class EspeakPhonemizer(Phonemizer):

    def __init__(self):
        self.logger = get_logger()
        lang = settings.ESPEAK_LANGUAGE
        if lang == "English":
            g2p_type = "espeak_ng_english_us_vits"
        else:
            g2p_type = f"attest_espeak_ng_{TO_LANGUAGE_CODE[lang]}"
        self.phoneme_tokenizer = PhonemeTokenizer(g2p_type)

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{1.name}}/g2p/espeak_phonemizer/phonemes.txt",
        method="txt",
        validator=validate_matching_to_project_size,
    )
    def phonemize_project(self, project: Project):
        phonemes = []

        self.logger.info("Phonemizing texts...")
        phonemes = [self.phonemize(text) for text in project.texts]

        self.logger.info("Phonemization is done, caching...")
        return phonemes

    @CacheHandler(
        cache_path_template=f"{settings.CACHE_DIR}/${{2}}",
        method="txt",
        validator=validate_matching_to_project_size,
    )
    def phonemize_many(self, texts: List[str], cache_path: str):
        phonemes = []

        for text in texts:
            phonemes.append(self.phonemize(text))
        self.logger.info("Phonemization is done!")

        return phonemes

    def phonemize(self, text):
        phonemes = self.phoneme_tokenizer.text2tokens(text)
        phonemes = "".join(phonemes).replace("<space>", " ")
        return phonemes
