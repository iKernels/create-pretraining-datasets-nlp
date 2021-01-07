# create-pretraining-datasets-nlp
Create large pre-training datasets for NLP

## Install

Install with pip

```bash
pip install git+https://github.com/iKernels/create-pretraining-datasets-nlp.git --upgrade
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

## How to use

Using a dataset created with the command before is very easy since it is a python dictionary with some enhancements under the hood. The only requirement is that keys must be integers (`int32`). There are two layers of compression: all values of the dictionary are individually compressed and each dump is finally compressed on disk.

You can load it from the dump file and use it as in the following example.
```python
>>> from create_pretraining_dataset.utils import CompressedDictionary
>>> 
>>> d = CompressedDictionary.load("/path/to/file.bz2")
>>> # OR
>>> d = CompressedDictionary.load("/path/to/file.xz", compression="xz")
>>> # OR
>>> d = CompressedDictionary()
>>> d[0] = {'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1], 'token_type_ids': [0, 0, 1, 1], 'words_tails': [True, False, True, True]}
>>>
>>> # use it like a normal dictionary
>>> # remember that keys are integers (to be better compatible with pytorch dataset indexing with integers)
>>> d[0]
{'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1], 'token_type_ids': [0, 0, 1, 1], 'words_tails': [True, False, True, True]}
>>>
>>> for k in d.keys():
>>>     # do something with d[k]
>>> # OR
>>> for k, value in d.items():
>>>     print(k, value) # print millions of entries is not always a good idea...
>>>
>>> # delete an entry
>>> del d[0]
>>>
>>> # get number of key-value pairs
>>> len(d)
1
>>>
>>> # access compressed data directly
>>> d.content[0]
b"3hbwuchbufbou&RFYUVGBKYU6T76\x00\x00" # some compressed byte array corresponding to the d[0] value
>>>
>>> # save the dict
>>> d.dump("/path/to/new/dump.bz2") # no compression argument. the compression is the same used for values.
```
