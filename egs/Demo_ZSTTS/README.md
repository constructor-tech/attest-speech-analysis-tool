# Zero-shot TTS demo project


These demonstration project was created by **[Constructor Technology](https://constructor.tech/)**.


Data:
  - LibriTTS test-clean, speakers 121 (female), 237 (female), 908 (male) and 7021 (male)
  - 5 random samples were used from each speaker (see Reference/meta/filelist.txt)
  - Additional 1 sample from each speaker were used as a prompt (samples 121_127105_000020_000004, 237_126133_000047_000000, 7021_85628_000002_000000 and 908_31957_000017_000001)


Models:

  - [HierSpeech](https://huggingface.co/spaces/LeeSangHoon/HierSpeech_TTS)
    (default settings)

  - [StyleTTS2](https://huggingface.co/spaces/styletts2/styletts2)
    (default settings)

  - [XTTSv2](https://huggingface.co/spaces/coqui/xtts)
    (default settings)


Results:

| Metric                            |    StyleTTS2 |    XTTSv2 |   HierSpeech++ |
|:----------------------------------|-------------:|----------:|---------------:|
| UTMOS ↑                           |    4.33261   |  3.91012  |       4.38437  |
| SpeechBERTScore ↑                 |    0.843454  |  0.81594  |       0.852341 |
| VDE ↓                             |   29.3502    | 28.5681   |      24.5089   |
| GPE ↓                             |   56.5696    | 58.8418   |      54.426    |
| FFE ↓                             |   66.3462    | 64.3822   |      61.5192   |
| logF0 RMSE ↓                      |    0.405351  |  0.41704  |       0.409221 |
| Squim STOI ↑                      |    0.997461  |  0.995439 |       0.998995 |
| Squim PESQ ↑                      |    3.66408   |  3.6414   |       3.9099   |
| Squim SI-SDR ↑                    |   24.8636    | 23.5334   |      26.1857   |
| Speaker Similarity (ECAPA-TDNN) ↑ |    0.377897  |  0.543946 |       0.529177 |
| CER ↓                             |    0.0267172 | 0.0117374 |      0.0229273 |
| WER ↓                             |    0.0562684 | 0.0221844 |      0.0541883 |
| PER ↓                             |    0.0146522 | 0.00313639|      0.0156468 |
| Character distance ↓              |    1.3       |  2.5      |       3.1      |
| Phoneme distance ↓                |    0.45      |  1.8      |       2        |