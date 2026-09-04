<html><head></head><body><h1>StyleBank</h1>

The **StyleBank** is a curated and structured collection of English texts gathered from multiple sources. It was developed in two phases. First, we devised a taxonomy of linguistic variation by drawing inspiration from previous theoretical work in this area. Second, we used this taxonomy to populate the StyleBank, identifying suitable corpora from prior work for each leaf category in the taxonomy.

## Taxonomy

To develop the taxonomy, we drew inspiration from influential work in sociolinguistics and linguistic variation. Based on this literature, we organized our taxonomy into four hierarchical levels.

At the highest level, the taxonomy distinguishes *Language variation* into *Individual* and *Intra-group* variations, corresponding respectively to variation associated with a specific speaker and variation shared by a community of speakers. The next level contains the principal dimensions of variation. Individual variation is represented by *Idiolect*, referring to linguistic characteristics associated with a particular individual. Intra-group variation is subdivided into *Diachronic* (variation across time), *Diatopic* (variation across geographical regions), *Diastratic* (variation across social groups), *Diaphasic* (variation across genres and registers), and *Diamesic* (variation across communication modalities).

Each of these dimensions is further refined into increasingly specific categories, culminating in a set of 83 leaf nodes that correspond to concrete linguistic varieties.

```text
Language variation
├── Individual
│   └── Idiolect
│       ├── Rihanna
│       ├── Shakira
│       └── [...]
│
└── Intra-group
    ├── Diachronic
    │   ├── Old English
    │   ├── Middle English
    │   └── [...]
    │
    ├── Diatopic
    │   ├── Indian English
    │   ├── Kenyan English
    │   └── [...]
    │
    ├── Diastratic
    │   ├── Age: 55--74
    │   ├── Education: Bachelor
    │   └── [...]
    │
    ├── Diaphasic
    │   ├── Informational
    │   ├── Opinion
    │   └── [...]
    │
    └── Diamesic
        ├── Digital
        └── Spoken
```

## Data

The 83 leaf nodes of the taxonomy form the basis of the text collection included in the StyleBank. Their inclusion was guided by the availability of suitable corpora corresponding to each category. To populate the StyleBank, we collected texts from the corpora listed in the table below.

Each leaf node is populated with between 10 and 64 texts (mean: 43 texts per leaf). In total, the StyleBank contains 3,586 texts. All texts in the StyleBank are between 8 and 168 words in length (mean: 36 words). We manually checked all texts to assess their quality.

| Variation Dimension | Leaf Node            | Resource                                                                                                                            |
| ------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Idiolect            | All                  | [Tweets Dataset](https://www.kaggle.com/datasets/mmmarchetti/tweets-dataset/data)                                                   |
| Diachronic          | Old English          | [NPEGL](https://spraakbanken.gu.se/en/resources/npegl-eng)                                                                          |
| Diachronic          | Middle English       | [Middle English Corpus](https://github.com/BenLambright/Middle-English-to-Modern-English-NMT)                         |
| Diachronic          | Early Modern English | [TCP Corpus](https://huggingface.co/datasets/uwgraphics/VEP2_TCP_SimpleText)                                                        |
| Diachronic          | Modern English       | [English Philosophical Texts](https://github.com/earlytexts/english-philosophical-texts)                               |
| Diatopic            | All                  | [eWAVE](https://github.com/cldf-datasets/ewave/blob/master/cldf/examples.csv)                                                                                                   |
| Diastratic          | All                  | [PASTEL](https://github.com/dykang/PASTEL)                                                                             |
| Diaphasic           | All                  | [CORE Corpus](https://github.com/TurkuNLP/CORE-corpus)                                                                 |
| Diamesic            | Digital              | [Reddit Post Comment Dataset](https://github.com/ishandandekar/Reddit_Post_Comment_Dataset)                            |
| Diamesic            | Spoken               | [This American Life Transcripts](https://www.kaggle.com/datasets/shuyangli94/this-american-life-podcast-transcriptsalignments/data) |
