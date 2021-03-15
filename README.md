# create-pretraining-datasets-nlp
Create large pre-training datasets for NLP


## Install

```bash
pip install git+https://github.com/iKernels/create-pretraining-datasets-nlp.git --upgrade
```


## Usage

List available datasets and other info about parameters
```bash
python -m create_pretraining_dataset --help
```

Create a `wikipedia` dataset (as a compressed dictionary) with:
```bash
python -m create_pretraining_dataset \
    --compression bz2   
    --strategy Electra \
    --output-file wikipediaen-electra-small-gen-128 \
    --name wikipedia 20200501.en \
    --tokenizer google/electra-small-generator \
    --max-sequence-length 128 \
    --processes 8 \
    --reduce 10
```

This example will read `1/10`th of the documents from the english wikipedia and will create a tokenized dataset with sentence pairs of about `128` tokens using `8` CPU cores. Datasets are taken from the [`datasets`](https://huggingface.co/docs/datasets/) library and tokenizers from the [`transformers`](https://huggingface.co/transformers/) repository. `CompressedDictionary`s are a special dictionaries taken from [repository](https://github.com/lucadiliello/compressed-dictionary). They allow for `dict`-like behaviour but compressing/decompressing values automatically to save memory.


### Other arguments

Datasets are created by concatenating sentences coming from the same document and possibly inserting a separator in the middle, just to make the model understand that a separator is. Each example is fill with sentences up to the maximum length (expressed in number of tokens, not words) and sometimes it is truncated just to accustom the model that sometimes sentences are truncated.

Available args:

- `-o` or `--output_file`:  Output file path.
- `--name`: Dataset to be parsed and preprocessed. Separate with a space the specific dataset name from the config, for example `wikipedia 20200501.en` or simply `openwebtext`. See all by running with `--help`.
- `--processes`: Number of parallel processes to use. Defaults to number of `#CPUs`.
- `--dataset_columns`: Columns names of the dataet to be used. Defaults to `['text']`.
- `--tokenizer`: Name or path of the tokenizer to be used to tokenize the text.
- `--max_sequence_length`: Max sequence length to fill examples. Defaults to `128`.
- `--do_not_pad`: Do not pad examples to `--max_sequence_length`.
- `--reduce`: Reduce number of documents in input by this quantity (useful for debugging).
- `--compression`': Compression algorithm of the output dictionary. Defaults to `bz2` (bzip2). Available also compression with `gzip` and `xz` (lzma2).
- `--strategy`: Which strategy should be used to create examples. See all by running with `--help`.
- `--compute_words_tails`. This will generate and additional boolean vector where a `True` means that the corresponding token is tail. A tail is a token in a word (composed of at least 2 tokens) that is not first. Useful when the model uses `whole word masking`.
- `--seed`: Set seed for reproducibility.


## Strategies

### ELECTRA

Pair sentencens with the following probabilities (that can be set from the command line).
- `--probability_random_length`: Probability of creating a sample with the first part (before the separator) having a random length between 5 and `max_sequence_length`. Defaults to `0.05`.
- `--probability_single_sentence`: Probability of creating an example containing a single sentence. Deafults to `0.1`.
- `--probability_first_segment_over_length`: Probability of creating a very longer first sequence, eventually truncated. Defaults to `0.5`. 


### BERT

Coming soon.


### RoBERTa

Coming soon.