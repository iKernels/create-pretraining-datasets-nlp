import random
import logging
import multiprocessing
import os
from argparse import ArgumentParser

from tqdm import tqdm
import datasets
from transformers import AutoTokenizer

from create_pretraining_dataset.utils import (multiprocessing_create_examples,
                   dataset_to_sentences, MODES, multiprocessing_addition)
from compressed_dictionary import CompressedDictionary

logging.getLogger().setLevel(logging.INFO)
ALL_DATASET_NAMES = datasets.list_datasets()


def heavy_load(dataset):
    """ Load entire dataset in memory. """
    res = list(tqdm(dataset['train'], desc="Loading entire dataset in memory", total=dataset['train'].num_rows))
    return res


def main(args):

    random.seed(args.seed)

    logging.info("Doing arguments checks")

    assert not os.path.isfile(args.output_file) or args.force_overwrite, (
        f"Output file {args.output_file} do already exist! Use `-f` to overwrite"
    )
    if os.path.isfile(args.output_file):
        os.remove(args.output_file)

    parsed_names = []
    for dataset_name in args.dataset_names:
        name, config = None, None

        dataset_name_splits = dataset_name.split(':')
        assert len(dataset_name_splits) > 0 and len(dataset_name_splits) < 3, (
            f"Found incorrect parameter in `--dataset-names`: {dataset_name}. Must be of the form `name` or `name:config`"
        )

        name = dataset_name_splits[0]
        if len(dataset_name_splits) > 1:
            config = dataset_name_splits[1]

        assert name in ALL_DATASET_NAMES, (
            f"dataset {name} is not available. Use `--help` to see all available datasets"
        )
        parsed_names.append((name, config))

    # this will check if args.tokenizer is available as path or pre_trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logging.info(f"Loaded tokenizer. Is it fast? {tokenizer.is_fast}")

    assert args.max_sequence_length is not None or args.max_sequence_length >= 0, (
        "`max-sequence-length` must be None or a positive integer"
    )

    assert args.limit is None or args.limit >= 0, (
        "`limit` must be None or a positive integer"
    )

    assert args.probability_random_length >= 0 and args.probability_random_length <= 1, (
        "`probability-random-length` must be None or a positive integer"
    )

    assert args.probability_single_sentence >= 0 and args.probability_single_sentence <= 1, (
        "`probability-single-sentence` must be None or a positive integer"
    )

    assert args.probability_first_segment_over_length >= 0 and args.probability_first_segment_over_length <= 1, (
        "`probability-first-segment-over-length` must be None or a positive integer"
    )

    final_cdictionary = CompressedDictionary()
    
    for name, config in parsed_names:     

        logging.info(f"Loading input dataset {name}")   
        dataset = datasets.load_dataset(name, config)
        logging.info(f"Processing input dataset {name} with {dataset['train'].num_rows} {'documents' if args.dataset_structure is MODES[0] else 'sentences'}")

        if args.all_in_memory:
            documents = heavy_load(dataset)
        else:
            documents = dataset['train']

        sentences = dataset_to_sentences(
            documents,
            limit=args.limit,
            mode=args.dataset_structure,
            limit_sentences_per_doc=args.sentences_per_doc
        )
        # 6000 - 7000 it/s with 8 threads

        examples = multiprocessing_create_examples(
            sentences,
            tokenizer,
            max_sequence_length=args.max_sequence_length,
            do_not_pad=args.do_not_pad,
            add_words_tails=args.compute_words_tails,
            probability_random_length=args.probability_random_length,
            probability_single_sentence=args.probability_single_sentence,
            probability_first_segment_over_length=args.probability_first_segment_over_length,
            num_processes=max(args.num_processes * 2 // 3, 1)
        )
        # 700 - 800 it/s with 8 threads

        multiprocessing_addition(
            final_cdictionary,
            examples,
            num_processes=max(args.num_processes // 3, 1),
            compression=args.compression
        ) # 500 - 600 it/s with 8 threads

    logging.info(f"Writing results to file {args.output_file}")
    final_cdictionary.dump(args.output_file)


if __name__ == "__main__":

    parser = ArgumentParser("Create a tokenized dataset for pre-training")

    parser.add_argument('-o', '--output-file', type=str, required=True, help="Output file path.")
    parser.add_argument('--dataset-names', type=str, required=True, nargs='+',
                        help=f"List of datasets to be parsed and built into the dataset."
                             f" Separate with a semicolon the specific dataset name from the config,"
                             f" for example `wikipedia:20200501.en`. Available datasets {ALL_DATASET_NAMES}")
    parser.add_argument('--tokenizer', required=True, type=str,  
                        help="Name of the tokenizer to use to tokenizer the text.")
    parser.add_argument('--max-sequence-length', type=int, required=True,
                        help="Max sequence length to fill sentence.")
    parser.add_argument('--num-processes', type=int, required=False, default=multiprocessing.cpu_count(),
                        help="Number of parallel processes to use.")
    parser.add_argument('--do-not-pad', action="store_true", help="Avoid padding to `max-sequence-length`.")
    parser.add_argument('--limit', type=int, required=False, default=None,
                        help='Limit number of documents in input.')
    parser.add_argument('--compression', type=str, required=False, default='bz2', choices=CompressedDictionary.ALLOWED_COMPRESSIONS,
                        help='Compression algorithm of the output dictionary.')
    parser.add_argument('-f', '--force-overwrite', action="store_true",
                        help='Overwrite output file if it does already exist.')
    parser.add_argument('--probability-random-length', default=0.05, required=False, type=float,
                        help="Probability of creating a sample with a random length between 5 and `max_sequence_length`.")
    parser.add_argument('--probability-single-sentence', default=0.1, required=False, type=float,
                        help="Probability of creating a sentence with a single sentence.")
    parser.add_argument('--probability-first-segment-over-length', default=0.5, required=False, type=float,
                        help="Probability of creating a longer first sequence.")
    parser.add_argument('--dataset-structure', default='one-doc-per-line', required=False, type=str,
                        choices=MODES,
                        help="Probability of creating a longer first sequence.")
    parser.add_argument('--sentences-per-doc', default=None, required=False, type=int,
                        help="If no empty line is found to separate documents when using `one-sentence-per-line`, use this maximal length (in number of sentences).")
    parser.add_argument('--compute-words-tails', action="store_true",
                        help="Words tails in an additional array in which True mean that the corresponding token is part of a composed word and is not the first. This array helps in creating whole word masks.")
    parser.add_argument('--seed', default=1337, required=False, type=int,
                        help="Seed for reproducibility.")
    parser.add_argument('--all-in-memory', action="store_true",
                        help="Seed for reproducibility.")

    args = parser.parse_args()
    main(args)
