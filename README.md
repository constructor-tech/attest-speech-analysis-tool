# ATTEST: an Analytics Tool for the Testing and Evaluation of Speech Technologies

**[Constructor Technology](https://constructor.tech/)**

**ATTEST** is a powerful evaluation framework designed to streamline the analysis of (synthesized) speech by integrating a variety of metrics across multiple dimensions. It consolidates speech evaluation into five distinct categories, each equipped with a set of metrics to thoroughly assess various aspects of speech quality:

- **Speech Intelligibility**: Focuses on how accurately a TTS model reproduces the intended text, emphasizing the clarity with which spoken words are understood. The primary metrics used include CER (Character Error Rate), WER (Word Error Rate), and PER (Phoneme Error Rate).
- **Speech Prosody**: Assesses the naturalness and expressiveness of speech prosody, using pitch analysis metrics such as voicing decision error (VDE), gross pitch error (GPE), fine pitch error (FFE), and logarithmic frequency root mean square error (Log F0 RMSE). ATTEST supports multiple pitch extraction engines, including Parselmouth, PyWorld, and CREPE.
- **Speaker Similarity**: Measures how closely the synthesized voice matches the target speaker, crucial for applications like voice cloning. Metrics for this include speaker similarity based on a comparison of embeddings obtained using the ECAPA-TDNN speaker verification model.
- **Signal Quality**: Analyzes the overall audio quality and intelligibility of the speech signal with metrics like PESQ, STOI, and TorchAudio-Squim.
- **MOS Prediction**: Uses metrics such as UTMOS and SpeechBERTScore to predict Mean Opinion Scores, simulating subjective listening tests through objective analysis.

By organizing metrics into five categories, ATTEST makes it easier to assess different qualities of speech. This structure helps developers see where a TTS model performs well and where it needs improvement, providing insights for further refinement.


## Contents

- [ATTEST: an Analytics Tool for the Testing and Evaluation of Speech Technologies](#attest-an-analytics-tool-for-the-testing-and-evaluation-of-speech-technologies)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Installation Notes](#installation-notes)
    - [Extra Packages](#extra-packages)
  - [Getting Started](#getting-started)
  - [ATTEST Overview](#attest-overview)
    - [Methods](#methods)
    - [Project and Group](#project-and-group)
    - [Feature Types](#feature-types)
    - [Metrics](#metrics)
    - [Attributes](#attributes)
    - [Language Support](#language-support)
  - [Advanced Usage: CLI Examples](#advanced-usage-cli-examples)
  - [Use Cases](#use-cases)
  - [Contributions](#contributions)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
    - [Third-Party Licenses](#third-party-licenses)
  - [Citing](#citing)


## Installation

Before installing, check the [extra packages](#extra-packages) section as you may want to expand the default requirements.

Set up a local environment using Python 3.10:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Installation Notes

1. **PyTorch Installation**: If you encounter issues with installing PyTorch, please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

2. **Espeak Phonemizer**: To use the `espeak-phonemizer` backend for phonemization, you need to install the `espeak-ng` system dependency. Detailed installation instructions are available in the [Phonemizer installation guide](https://bootphon.github.io/phonemizer/install.html#dependencies).


### Extra Packages

1. **Nemo Text Normalization for CER, WER, PER, Character Distance, and Phoneme Distance**: Install `nemo-text-processing==0.2.2rc0`. By default, text normalization is not performed to compute metrics that rely on text comparison. If you encounter issues on MacOS or Windows, refer to the [official installation guide](https://github.com/NVIDIA/NeMo-text-processing?tab=readme-ov-file#installation).


## Getting Started

To start the application with a user interface (UI), use the following command:

```bash
python3 run.py ui
```


## ATTEST Overview


### Methods

ATTEST offers three primary methods for evaluation and analysis:

1. **Evaluate**: Use this method to analyze a single project in detail. It provides both a general overview and detailed information on individual examples.
   
2. **Compare**: Use this method for comparing two projects side by side. It provides both a general overview and detailed information on individual examples.
   
3. **Multiple Compare**: Use this method for comparing several projects at once. It provides an overview of the selected metrics in both table and graph formats.


### Project and Group

In the context of ATTEST, a "project" refers to a dataset containing real or synthetic recordings. Each project is organized into two main folders:

  - **meta**: Contains a filelist.txt file, which describes recordings. This file has two columns separated by a | symbol:

    - The first column provides the relative path to the audio file within the wavs folder.
    - The second column contains the text that is spoken in the corresponding audio file.

  - **wavs**: Contains the actual audio files. These files should be in WAV format with a single audio channel (mono) and a sampling rate of 22050 Hz.

Projects can be organized into groups. For example, in the [egs/Demo_ZSTTS](egs/Demo_ZSTTS) directory, Demo_ZSTTS serves as the group name, containing multiple related projects.

### Feature Types

ATTEST provides a variety of features to evaluate different aspects of speech quality. These features are categorized into two types:

 - **Metric**: A numerical value computed from the sample or by comparing the sample with a reference sample.
 - **Attribute**: A property extracted from the sample. It can be text, audio, an image, a numerical value, or a more complex object.

You can enable and disable features on the sidebar in the UI. You can choose what features are enabled by default in the [attest/config/config.yaml](attest/config/config.yaml).


### Metrics

Below is a list of the metrics available in ATTEST. Each metric also has an identifier that can be used with the [command line interface (CLI)](#advanced-usage-cli-examples). Some metrics require a reference sample. Some metrics could be accelerated with the use of a GPU, as indicated.

1. **MOS Prediction**

    - **UTMOS** *(CLI Identifier: `utmos`, GPU preferred)*: Predicts the Mean Opinion Score (MOS) to assess the overall perceived quality of synthesized speech.
    - **SpeechBERTScore**  *(CLI Identifier: `speech_bert_score`, Reference required, GPU preferred)*: Measures the similarity between synthesized and reference speech by comparing their contextualized embeddings derived from a WavLM model.

2. **Speech intelligibility**

    - **CER (Character Error Rate)** (CLI Identifier: `cer`, GPU preferred): The percentage of characters that were incorrectly predicted by the Whisper speech recognition model compared to the original text.
    - **WER (Word Error Rate)** (CLI Identifier: `wer`, GPU preferred): The percentage of words that were incorrectly predicted by the Whisper speech recognition model compared to the original text.
    - **PER (Phoneme Error Rate)** (CLI Identifier: `per`, GPU preferred): The percentage of phonemes that were incorrectly predicted, calculated by using the Whisper speech recognition model and grapheme-to-phoneme (G2P) conversion to compare the phonemes of the original text and the transcription.
    - **Character distance** *(CLI Identifier: `character_distance`, GPU preferred)*: The number of distinct symbols between the original text and the transcription obtained from Whisper speech recognition model.
    - **Phoneme distance** *(CLI Identifier: `phoneme_distance`, GPU preferred)*: The number of distinct phonemes between the original text and the transcription obtained from Whisper speech recognition model.

3. **Speech intonation**

    - **VDE** *(CLI Identifier: `vde`, Reference required, GPU preferred if used torchcrepe model for the extraction)*: Voicing decision error from [Reducing F0 Frame Error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend](https://www.seas.ucla.edu/spapl/paper/chu_icassp_09.pdf)
    - **GPE** *(CLI Identifier: `gpe`, Reference required, GPU preferred if used torchcrepe model for the extraction)*: Gross Pitch Error  from [Reducing F0 Frame Error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend](https://www.seas.ucla.edu/spapl/paper/chu_icassp_09.pdf)
    - **FFE** *(CLI Identifier: `ffe`, Reference required, GPU preferred if used torchcrepe model for the extraction)*: F0 Frame Error  from [Reducing F0 Frame Error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend](https://www.seas.ucla.edu/spapl/paper/chu_icassp_09.pdf)
    - **logF0 RMSE** *(CLI Identifier: `logf0_rmse`, Reference required, GPU preferred if used torchcrepe model for the extraction)*: Computes the root mean square error of the logarithmic fundamental frequency (F0) between synthesized and reference speech.

4. **Signal quality**

    - **Squim STOI** *(CLI Identifier: `squim_stoi`, GPU preferred)*: Reference-free estimation of Wideband Perceptual Estimation of Speech Quality (PESQ) from [Torchaudio](https://pytorch.org/audio/stable/tutorials/squim_tutorial.html).
    - **Squim PESQ** *(CLI Identifier: `squim_pesq`, GPU preferred)*: Reference-free estimation of Short-Time Objective Intelligibility (STOI) from [Torchaudio](https://pytorch.org/audio/stable/tutorials/squim_tutorial.html).
    - **Squim SI-SDR** *(CLI Identifier: `squim_sisdr`, GPU preferred)*: Reference-free estimation of Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) from [Torchaudio](https://pytorch.org/audio/stable/tutorials/squim_tutorial.html).

5. **Speaker similarity**

    - **Speaker Similarity (ECAPA-TDNN)** *(CLI Identifier: `sim_ecapa`, Reference required, GPU preferred)*: Calculates the similarity between synthesized and reference voices using the ECAPA-TDNN speaker verification model to assess how closely the synthesized voice matches the target speaker.

### Attributes

Below is a list of the attributes available in ATTEST. Each attribute also has an identifier that can be used with the [command line interface (CLI)](#advanced-usage-cli-examples).

- **Audio** *(CLI Identifier: `audio`)*: The actual audio waveform.
- **Text** *(CLI Identifier: `text`)*: The original written text intended to be synthesized or spoken in the audio.
- **Text (normalized)** *(CLI Identifier: `text_norm`)*: The processed version of the text where the text normalization method is specified in the config (for CLI) or the settings tab (UI).
- **Text phonemes** *(CLI Identifier: `text_phonemes`)*: The phonetic transcription of the text.
- **Transcript** *(CLI Identifier: `transcript`)*: The text derived from ASR.
- **Transcript phonemes** *(CLI Identifier: `transcript_phonemes`)*: The phonetic transcription of the ASR-generated transcript.
- **Grapheme pronunciation speed** *(CLI Identifier: `pronunciation_speed`)*: The rate at which characters or letters are spoken in the audio, measured in character units per second.
- **Phoneme pronunciation speed** *(CLI Identifier: `pronunciation_speed_phonemes`)*: The pronunciation rate, measured in phoneme units per second.
- **Audio duration** *(CLI Identifier: `audio_duration`)*: The duration of the actual audio waveform.
- **Speech duration** *(CLI Identifier: `speech_duration`)*: The duration of actual spoken content within the audio, excluding silence at the beginning and at the end.
- **Silence in the begining** *(CLI Identifier: `silence_begin`)*: The duration of silence at the beginning.
- **Silence in the end** *(CLI Identifier: `silence_end`)*: The duration of silence at the end.
- **Pitch mean** *(CLI Identifier: `pitch_mean`, GPU preferred if used torchcrepe model for the extraction)*: The average fundamental frequency (F0) across the speech sample, indicating the overall pitch level.
- **Pitch std** *(CLI Identifier: `pitch_std`, GPU preferred if used torchcrepe model for the extraction)*: The standard deviation of the fundamental frequency, reflecting pitch variability.
- **Pitch plot** *(CLI Identifier: `pitch_plot`, GPU preferred if used torchcrepe model for the extraction)*: A visual representation of the pitch contour over time, illustrating intonation patterns and prosody.
- **Wavelet prosody plot** *(CLI Identifier: `wavelet_prosody`)*: A graphical depiction of prosodic features like pitch, energy, and wavelets, generated using the [wavelet_prosody_toolkit](https://github.com/asuni/wavelet_prosody_toolkit) project.


### Language Support

ATTEST provides metrics that vary in language compatibility:

- **Language-idependent metrics**: Metrics such as VDE, GPE, FFE, logF0 RMSE, and Squim family metrics (STOI, PESQ, SI-SDR) are language-independent, as they reflect properties unrelated to specific languages.
- **Applicable to all languages**: Metrics like UTMOS, SpeechBERTScore, Speaker Similarity (ECAPA-TDNN) and Squim-metrics use components trained primarily on English data. However, since the metric reflects a language-independent property, it could generalize to audio in other languages.
- **Language-specific metrics**:
  - CER, WER, Character distance: Limited to languages supported by [Whisper](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10).
  - PER, Phoneme distance: Limited to languages supported by both [Whisper](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10) and [Espeak](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md).

Refer to the [metrics table](docs/metrics_overview.md) for a detailed view of language compatibility.


## Advanced Usage: CLI Examples

This section provides detailed examples of how to use the ATTEST CLI for specific tasks.

1. **Evaluate** a single project using specific features:

    ```bash
    python3 run.py evaluate --project <your_project> --features <first feature> <second feature> ... <last feature> [--output <output_file>]
    ```

    Replace `<your_project>` with the path to your project and `<first feature>`, `<second feature>`, etc., with the names of the features you want to evaluate. If the `--output` option is specified, the results will be saved in JSON format to the given file.

    **Example**: You can evaluate the egs/Demo_ZSTTS/StyleTTS2 project using the UTMOS metric as follows:

    ```bash
    python3 run.py evaluate --project egs/Demo_ZSTTS/StyleTTS2 --features utmos
    ```


2. **Compare** two projects by specifying the features to be evaluated:

    ```bash
    python3 run.py compare --project1 <project_1> --project2 <project_2> --features <first feature> <second feature> ... <last feature> [--output <output_file>]
    ```

    Replace `<project_1>` and `<project_2>` with the paths to your projects and `<first feature>`, `<second feature>`, etc., with the names of the features you want to compare. If the `--output` option is specified, the results will be saved in JSON format to the given file.

    **Example**: You can compare the project egs/Demo_ZSTTS/StyleTTS2 with the reference project egs/Demo_ZSTTS/Reference using the UTMOS and Speaker Similarity metrics and save the results to `compare_results.json` as follows:

    ```bash
    python3 run.py compare --project1 egs/Demo_ZSTTS/Reference --project2 egs/Demo_ZSTTS/StyleTTS2 --features utmos sim_ecapa  --output compare_results.json
    ```

3. **Compare multiple** projects simultaneously across various features:

    ```bash
    python3 run.py multiple_compare --projects <project_1> <project_2> ... <last project> --features <first feature> <second feature> ... <last feature> [--output <output_file>]
    ```

    Replace `<project_1>`, `<project_2>`, etc., with the paths to your projects and `<first feature>`, `<second feature>`, etc., with the names of the features you want to compare. If the `--output` option is specified, the results will be saved in JSON format to the given file.

    **Example**: You can compare the projects egs/Demo_ZSTTS/StyleTTS2 egs/Demo_ZSTTS/XTTSv2 with the reference project egs/Demo_ZSTTS/Reference using the UTMOS and Speaker Similarity metrics as follows:

    ```bash
    python3 run.py multiple_compare --projects egs/Demo_ZSTTS/Reference egs/Demo_ZSTTS/StyleTTS2 egs/Demo_ZSTTS/XTTSv2 --features utmos sim_ecapa
    ```


## Use Cases

1. **Comparing multiple projects for reports**:
   - Use the UI multiple compare method and export results in CSV, LaTeX, or Markdown format. This method also displays histograms for each metric, showing the mean values of project scores, which can be included in your reports as well for a visual representation of the performance differences.

2. **Analyzing audio quality of a single project**:
   - Use the UI evaluate and compare methods (if reference recordings are available). It is possible to sort the results based on a specific metric or even sort by the difference between two metrics. This can help identify where two models deviate from each other, providing insights into specific areas of performance differences.

3. **Processing a large number of projects (e.g. comparison of different speech prompts for ZSTTS)**:
   - Use CLI commands to compute metrics and get results in JSON format.

4. **Filtering datasets for TTS model training**:
   - Compute metrics one-by-one using CLI commands.


## Contributions

We welcome contributions to improve ATTEST! Whether you're fixing bugs, adding new features, adding new benchmarks, or improving documentation, your help is appreciated.

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them.
4. Submit a pull request with a detailed explanation of your changes.


## Acknowledgements

ATTEST is built upon and integrates various tools, libraries, and models. We would like to acknowledge the following projects for their contributions:

- **[Streamlit](https://github.com/streamlit/streamlit)**: Powers the user interface.
- **[wavelet_prosody_toolkit](https://github.com/asuni/wavelet_prosody_toolkit)**: Used for generating wavelet prosody plots.
- **[Discrete Speech Metrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics)**: Utilized for the SpeechBERTScore metric.
- **[UTMOS](https://github.com/sarulab-speech/UTMOS22)**: Used for the UTMOS metric.
- **[Whisper](https://github.com/openai/whisper)**: Used as the ASR and forced alignment engine.
- **[WavLM](https://github.com/microsoft/unilm)**: Contributes to the SpeechBERTScore metric.
- **[SpeechBrain](https://github.com/speechbrain/speechbrain)**: Used for speaker similarity using the ECAPA-TDNN model.
- **[OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer)**: Used as the grapheme-to-phoneme (G2P) engine.
- **[Phonemizer](https://github.com/bootphon/phonemizer)**: Used as the grapheme-to-phoneme (G2P) engine.
- **[torchcrepe](https://github.com/maxrmorrison/torchcrepe)**: Used as the pitch extraction engine.
- **[Parselmouth](https://github.com/YannickJadoul/Parselmouth)**: Used as the pitch extraction engine.
- **[PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)**: Used as the pitch extraction engine.
- **[VDE, GPE, FFE, logF0 RMSE](https://www.seas.ucla.edu/spapl/paper/chu_icassp_09.pdf)**: Used as speech intonation metrics.


## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for the full license text.

### Third-Party Licenses

This project uses third-party libraries and code, which are distributed under their respective licenses. A list of these dependencies and their licenses can be found in the [NOTICE](./NOTICE) file.


## Citing

If you find our work is useful in your research, please cite the following paper:

```
@inproceedings{obukhov24_interspeech,
  title     = {ATTEST: an analytics tool for the testing and evaluation of speech technologies},
  author    = {Dmitrii Obukhov and Marcel {de Korte} and Andrey Adaschik},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {3646--3647},
}
```
