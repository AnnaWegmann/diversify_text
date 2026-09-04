# StyleBank

Code to build the **StyleBank**.

## Reproduction

### 1. Download the source files

Download the missing source files required by the individual corpus extraction scripts and place them in the appropriate corpus directories. Most source files are already included in the related directories (cf. `.gitignore`).

The required source files can be downloaded from the following resources:

| Corpus                                 | Source                                                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Tweets Dataset (`tweets`)              | [Kaggle](https://www.kaggle.com/datasets/mmmarchetti/tweets-dataset/data)                                   |
| NPEGL (`korp`)                         | [Språkbanken](https://spraakbanken.gu.se/en/resources/npegl-eng)                                            |
| Middle English Corpus (`chaucer`)      | [GitHub](https://github.com/BenLambright/Middle-English-to-Modern-English-NMT)                              |
| eWAVE (`ewave`)                        | [GitHub](https://github.com/cldf-datasets/ewave/blob/master/cldf/examples.csv)                              |
| PASTEL (`pastel`)                      | [GitHub](https://github.com/dykang/PASTEL)                                                                  |
| CORE Corpus (`core`)                   | [GitHub](https://github.com/TurkuNLP/CORE-corpus)                                                           |
| This American Life Transcripts (`tal`) | [Kaggle](https://www.kaggle.com/datasets/shuyangli94/this-american-life-podcast-transcriptsalignments/data) |

### 2. Extract corpus samples

Run each individual `extract_[corpus]_sample.py` script to extract the required sample from each corpus.

For example:

```bash
python extract_ewave_sample.py
```

The resulting samples will be stored inside the corresponding `ewave` folder.

### 3. Create the StyleBank

Once all corpus samples have been generated, run:

```bash
python create_stylebank.py
```

This generates the final:

```text
stylebank.json
```

This file additionally underwent a manual check.

### 4. Generate statistics (optional)

To generate statistics for the resulting StyleBank, run:

```bash
python stylebank_stats.py
```

This produces:

```text
stylebank_stats.json
```

The file contains various statistics describing the StyleBank.