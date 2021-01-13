# create-pretraining-datasets-nlp
Create large pre-training datasets for NLP


## Install

Install with pip

```bash
pip install git+https://github.com/iKernels/create-pretraining-datasets-nlp.git --upgrade
```

install also `compressed-dictionary` library
```bash
pip install git+https://github.com/lucadiliello/compressed-dictionary.git --upgrade
```


## Usage

List available datasets and other info about parameters
```bash
python -m create_pretraining_dataset --help
```

Create a dataset (as a compressed dictionary)
```bash
python -m create_pretraining_dataset \
    --output_file dataset.xz \
    --dataset-names wikipedia:20200501.en wikipedia:20200501.it \
    --tokenizer bert-base-cased \
    --max-sequence-length 128 \
    --num-processes 4 \
    --limit 20
```

This example will read the first `20` documents from both the italian and the english wikipedia and will create a tokenized dataset with sentence pairs of about `128` tokens of length using `4` CPU cores. Datasets are taken from the [`datasets`](https://huggingface.co/docs/datasets/) library and tokenizers from the [`transformers`](https://huggingface.co/transformers/) repository.

### Other arguments

Datasets are created by concatenating sentences coming from the same document and possibly inserting a separator in the middle, just to make the model understand that a separator is. Each example is fill with sentences up to the maximum length (expressed in number of tokens, not words) and sometimes it is truncated just to accustom the model that sometimes sentences are truncated.

Available args:

- `-o` or `--output-file`:  Output file path.
- `--dataset-names`: List of datasets to be parsed and built into the dataset. Separate with a semicolon the specific dataset name from the config, for example `wikipedia:20200501.en`
- `--tokenizer`: Name or path of the tokenizer to be used to tokenize the text.
- `--max-sequence-length`: Max sequence length to fill examples. Defaults to `128`.
`--num-processes`: Number of parallel processes to use. Defaults to number of `#CPUs - 2`.
- `--do-not-pad`: Do not pad examples to `max-sequence-length`.
- `--limit`: Limit number of documents in input for each dataset.
- `--compression`': Compression algorithm of the output dictionary. Defaults to `bz2` (bzip2). Available also compression with `gzip` and `xz` (lzma2).
- `-f` or `--force-overwrite`: Overwrite output file if it does already exist.
- `--probability-random-length`: Probability of creating a sample with the first part (before the separator) having a random length between 5 and `max_sequence_length`. Defaults to `0.05`.
- `--probability-single-sentence`: Probability of creating an example containing a single sentence. Deafults to `0.1`.
- `--probability-first-segment-over-length`: Probability of creating a very longer first sequence, eventually truncated. Defaults to `0.5`. 
- `--dataset-structure`: How the dataset is structured. At the moment is provided support for `one-doc-per-line` and `one-sentence-per-line`. 
- `--sentences-per-doc`: Collect at most this number of sentences in one document when using `--dataset-structure=one-sentence-per-line`. This will apply in parallel with splitting of documents on empty lines. Defaults to `None`, that is, use only empty lines as documents separators. Be aware that some datasets like `bookcorpus` do not contain documents separators like empty lines.
`--compute-words-tails`. This will generate and additional boolean vector where a True means that the corresponding token is tail. A tail is a token in a word (composed of at least 2 tokens) that is not first. 
- `--seed`: Set seed for reproducibility.


## Useful examples

### Wikipedia

- Create wikipedia dataset pretokenized with the bert tokenizer `bert-base-cased` with a maximal sequence length of `128`.
```bash
python -m create_pretraining_dataset --compression bz2 \
    --output-file wikipedia-bert-cased-128-example.bz2 \
    --dataset-names wikipedia:20200501.en \
    --tokenizer bert-base-cased \
    --max-sequence-length 128 --num-processes 16
```

- Same but with a maximal sequence length of `512`.
```bash
python -m create_pretraining_dataset --compression bz2 \
    --output-file wikipedia-bert-cased-512-example.bz2 \
    --dataset-names wikipedia:20200501.en \
    --tokenizer bert-base-cased \
    --max-sequence-length 512 --num-processes 16
```

- Same but with the `roberta-base` tokenizer.
```bash
python -m create_pretraining_dataset --compression bz2 \
    --output-file wikipedia-roberta-512-example.bz2 \
    --dataset-names wikipedia:20200501.en \
    --tokenizer roberta-base \
    --max-sequence-length 512 --num-processes 16
```


### OpenWebText

- Create openwebtext dataset pretokenized with the bert tokenizer `bert-base-cased`.
```bash
python -m create_pretraining_dataset --compression bz2 \
    --output-file openwebtext-bert-cased-128-example.bz2 \
    --dataset-names openwebtext \
    --tokenizer bert-base-cased \
    --max-sequence-length 128 --num-processes 16
```

- Same but limiting total number of documents and increasing maximum sequence length.
```bash
python -m create_pretraining_dataset --compression bz2 \
    --output-file openwebtext-bert-cased-512-example.bz2 \
    --dataset-names openwebtext \
    --tokenizer bert-base-cased \
    --max-sequence-length 512 --num-processes 16 --limit 200
```


### BookCorpus

- Create bookcorpus with a maximum sequeunce length of `128`. Bookcorpus contains a sentence per line, instead of a document per line like the majority of the datasets.
```bash
python -m create_pretraining_dataset \
    --output-file bookcorpus-bert-cased-128.bz2 \
    --dataset-names bookcorpus \
    --tokenizer bert-base-cased \
    --max-sequence-length 128 \
    --dataset-structure one-sentence-per-line --sentences-per-doc 200
```

- Same but with a maximum sequeunce length of `512`.
```bash
python -m create_pretraining_dataset \
    --output-file bookcorpus-bert-cased-512.bz2 \
    --dataset-names bookcorpus \
    --tokenizer bert-base-cased \
    --max-sequence-length 512 \
    --dataset-structure one-sentence-per-line --sentences-per-doc 200
```


### BookCorpusOpen

- Create bookcorpus (open version) with a maximum sequeunce length of `128`.
```bash
python -m create_pretraining_dataset \
    --output-file bookcorpusopen-bert-cased-128.bz2 \
    --dataset-names bookcorpusopen \
    --tokenizer bert-base-cased \
    --max-sequence-length 128
```

- Same but with a maximum sequeunce length of `512` and computing words tails.
```bash
python -m create_pretraining_dataset \
    --output-file bookcorpusopen-bert-cased-512.bz2 \
    --dataset-names bookcorpusopen \
    --tokenizer bert-base-cased \
    --max-sequence-length 512 --compute-words-tails
```


## CompressedDictionary

This piece of code has been moved in a separate [repository](https://github.com/lucadiliello/compressed-dictionary) for better code management.