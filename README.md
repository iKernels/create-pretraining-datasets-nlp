# create-pretraining-datasets-nlp
Create large pre-training datasets for NLP

## Install

Install with pip

```bash
pip install git+https://github.com/iKernels/create-pretraining-datasets-nlp.git --upgrade
```

## Usage

List available datasets
```bash
python -m create_dataset.pretraining --dataset-names-list
```

Create a dataset (as a compressed dictionary)
```bash
python -m create_dataset.pretraining \
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
- `--dataset-names`: List of datasets to be parsed and built into the dataset. Use `--dataset-names-list`. For more info about available datasets. Separate with a semicolon the specific dataset name from the config, for example `wikipedia:20200501.en`
- `--dataset-names-list`: List all available datasets.
- `--tokenizer`: Name or path of the tokenizer to be used to tokenize the text.
- `--max-sequence-length`: Max sequence length to fill examples.
`--num-processes`: Number of parallel processes to use. Defaults to number of `#CPUs - 2`.
- `--do-not-pad`: Do not pad examples to `max-sequence-length`.
- `--limit`: Limit number of documents in input for each dataset.
- `--compression`': Compression algorithm of the output dictionary. Defaults to 'xz' (lzma). Available also compression with `gzip` and `bz2`.
- `-f` or `--force-overwrite`: Overwrite output file if it does already exist.
- `--probability-random-length`: Probability of creating a sample with the first part (before the separator) having a random length between 5 and `max_sequence_length`. Defaults to `0.05`.
- `--probability-single-sentence`: Probability of creating an example containing a single sentence. Deafults to `0.1`.
- `--probability-first-segment-over-length`: Probability of creating a very longer first sequence, eventually truncated. Defaults to `0.5`.
