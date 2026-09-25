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

This generates the first:

```text
stylebank.json
```

### 4. Changes after the first version.

We manually scanned, removed and edited the examples to make sure they are of high quality (i.e., no noise and representing the style in question).

We also made more additions: 

We run `extract_lyrical_sample.py` extract ALL 78 lyrical/prose texts across train/dev/test from https://huggingface.co/datasets/TajaKuzmanPungersek/X-GENRE-text-genre-dataset. Then starting from the first example, we manually went through the generated json file to fill up the `lyrical` leaf to 13 (some lyrics had been removed in the manual edits). Additionally, we created the `prose` leaf and filled it up to 10 as well starting from the first of the 78 example texts going until 10 high quality samples had been added.


### 5. Generate statistics (optional)

To generate statistics for the resulting StyleBank, run:

```bash
python stylebank_stats.py --input stylebank.json --output stylebank_stats.json
```

This produces:

```text
stylebank_stats.json
```

The file contains various statistics describing the StyleBank.