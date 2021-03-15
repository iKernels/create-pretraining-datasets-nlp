import random
import logging
import multiprocessing
import os
from argparse import ArgumentParser, Namespace

import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

from compressed_dictionary import CompressedDictionary
from transformers_lightning.utils import get_classes_from_module
from create_pretraining_dataset import strategies
from create_pretraining_dataset.strategies import _Strategy
from create_pretraining_dataset import __version__


logging.getLogger().setLevel(logging.INFO)
ALL_DATASET_NAMES = datasets.list_datasets()
ALL_STRATEGIES = get_classes_from_module(strategies, parent=_Strategy)


def main(args):

    logging.info("Setting seed...")
    random.seed(args.seed)

    logging.info("Checking args...")
    assert not os.path.isfile(args.output_file), (
        f"Output file {args.output_file} do already exist!"
    )

    name, config = None, None

    assert len(args.name) in [1, 2], (
        f"Found incorrect value for parameter `--name`: {args.name}. Must be of the form `<name>` or `--name <name> <config>`"
    )

    name = args.name[0]
    if len(args.name) > 1:
        config = args.name[1]

    assert name in ALL_DATASET_NAMES, (
        f"dataset {name} is not available. Use `--help` to see all available datasets"
    )

    # this will check if args.tokenizer is available as path or pre_trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logging.info(f"Loaded tokenizer. Is it fast? {tokenizer.is_fast}")

    assert args.max_sequence_length is not None or args.max_sequence_length >= 0, (
        "`max-sequence-length` must be None or a positive integer"
    )

    assert args.reduce is None or args.reduce >= 1, (
        "`reduce` must be None or a positive integer"
    )

    strategy = ALL_STRATEGIES[args.strategy](args, tokenizer)

    logging.info(f"Loading input dataset {name}")   
    dataset = datasets.load_dataset(name, config)['train']

    if args.reduce is not None:
        # filter away examples
        logging.info(f"Reducing dataset size by {args.reduce} times")
        dataset = dataset.shard(num_shards=args.reduce, index=0, contiguous=False)

    # process dataset
    logging.info(f"Processing input dataset {name} with {dataset.num_rows} documents")

    def process_fn(data):
        return { 'data': [CompressedDictionary.__compress__(v, compression=args.compression) for v in strategy(data)] }

    processed = dataset.map(
        function=process_fn,
        batched=True,
        load_from_cache_file=False,
        remove_columns=dataset.column_names,
        disable_nullable=True,
        input_columns=args.dataset_columns,
        num_proc=args.processes,
        writer_batch_size=10**4
    )
    logging.info(f"Dataset processed: {len(processed)} examples created")

    final_cdictionary = CompressedDictionary(compression=args.compression)
    logging.info(f"Creating compressed dictionary...")
    for i, data in tqdm(enumerate(processed), total=len(processed), desc="Adding to dictionary"):
        final_cdictionary.__add_already_compresses_value__(i, data['data'])

    logging.info(f"Writing results to file {args.output_file}")
    final_cdictionary.dump(args.output_file)


if __name__ == "__main__":

    parser = ArgumentParser("Create a tokenized dataset for pre-training")

    parser.add_argument('-o', '--output_file', type=str, required=True, help="Output file path.")
    parser.add_argument('--name', type=str, required=True, nargs='+',
                        help=f"Dataset name to be parsed and preprocessed."
                             f" Separate with a semicolon the specific dataset name from the config,"
                             f" for example `wikipedia:20200501.en`. Available datasets {ALL_DATASET_NAMES}")
    parser.add_argument('--processes', type=int, required=False, default=multiprocessing.cpu_count(),
                        help="Number of parallel processes to use.")
    parser.add_argument('--dataset_columns', nargs='+', default=['text'], required=False, type=str,
                        help="Columns names in the dataset. Provide many if nested.")
    
    parser.add_argument('--tokenizer', required=True, type=str,  
                        help="Name of the huggingface pre-trained tokenizer to use to tokenizer the text.")
    parser.add_argument('--max_sequence_length', type=int, required=True,
                        help="Max sequence length to fill sentence.")
    parser.add_argument('--do_not_pad', action="store_true", help="Avoid padding to `max-sequence-length`.")

    parser.add_argument('--reduce', type=int, required=False, default=None,
                        help='Limit number of documents in input reducing dataset size by this number.')
    parser.add_argument('--compression', type=str, required=False, default='bz2', choices=CompressedDictionary.ALLOWED_COMPRESSIONS,
                        help='Compression algorithm of the output compressed dictionary.')

    parser.add_argument('--strategy', type=str, required=True, choices=ALL_STRATEGIES, help="Strategy to use to create the dataset.")
    parser.add_argument('--compute_words_tails', action="store_true",
                        help="Words tails in an additional array in which True mean that the corresponding token is part of a composed word and is not the first. This array helps in creating whole word masks.")
    parser.add_argument('--seed', default=1337, required=False, type=int,
                        help="Seed for reproducibility.")

    # add strategy parameters
    tmp_args, _ = parser.parse_known_args()
    strategy_class = ALL_STRATEGIES[tmp_args.strategy]
    parser = strategy_class.add_arguments_to_argparse(parser)

    args = parser.parse_args()
    main(args)
