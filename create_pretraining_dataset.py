import logging
import multiprocessing
import os
from argparse import ArgumentParser

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import (CompressedDictionary, create_examples,
                   documents_to_sentences, multiprocessing_tokenizer,
                   prepare_datasets)

logging.getLogger().setLevel(logging.INFO)
ALL_DATASET_NAMES = datasets.list_datasets()


def main(args):
    
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
            "dataset {name} is not available. Use `--dataset-names-list` to see all available datasets"
        )
        parsed_names.append((name, config))

    # this will check if args.tokenizer is available as path or pre_trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logging.info(f"Loaded tokenizer. Is it fast? {tokenizer.is_fast}")

    assert args.max_sequence_length is not None or args.max_sequence_length >= 0, (
        "`max-sequence-length` must be None or a positive integer"
    )

    assert args.limit is not None or args.limit >= 0, (
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
    global_counter = 0

    for name, dataset in zip(parsed_names, prepare_datasets(parsed_names=parsed_names)):
        
        logging.info(f"Processing input dataset {name} with {dataset['train'].num_rows} documents")
        documents = dataset['train']['text']

        sentences = documents_to_sentences(documents, limit=args.limit)

        tokenizer_dataset_generator = multiprocessing_tokenizer(
            sentences, tokenizer=tokenizer, num_processes=args.num_processes
        )

        examples = create_examples(
            tokenizer,
            tokenizer_dataset_generator,
            max_sequence_length=args.max_sequence_length,
            do_not_pad=args.do_not_pad,
            probability_random_length=args.probability_random_length,
            probability_single_sentence=args.probability_single_sentence,
            probability_first_segment_over_length=args.probability_first_segment_over_length
        )

        for example in tqdm(examples, desc="Adding samples to compressed dictionary"):
            final_cdictionary[global_counter] = example
            global_counter += 1

    final_cdictionary.to_file(args.output_file)


if __name__ == "__main__":

    parser = ArgumentParser("Create a tokenized dataset for pre-training")

    parser.add_argument('-o', '--output-file', type=str, required=True, help="Output file path.")
    parser.add_argument('--dataset-names', type=str, required=True, nargs='+',
                        help="List of datasets to be parsed and built into the dataset."
                             "Use `--dataset-names-list` for more info about available datasets."
                             "Separate with a semicolon the specific dataset name from the config,"
                             " for example `wikipedia:20200501.en`")
    parser.add_argument('--dataset-names-list', action="store_true", help="List available datasets.")
    parser.add_argument('--tokenizer', required=True, type=str,  
                        help="Name of the tokenizer to use to tokenizer the text.")
    parser.add_argument('--max-sequence-length', type=int, required=False, default=None,
                        help="Max sequence length to fill sentence.")
    parser.add_argument('--num-processes', type=int, required=False, default=(multiprocessing.cpu_count() - 2),
                        help="Number of parallel processes to use.")
    parser.add_argument('--do-not-pad', action="store_true", help="Avoid padding to `max-sequence-length`.")
    parser.add_argument('--limit', type=int, required=False, default=None,
                        help='Limit number of documents in input.')
    parser.add_argument('--compression', type=str, required=False, default='xz', choices=CompressedDictionary.ALLOWED_COMPRESSIONS,
                        help='Compression algorithm of the output dictionary.')
    parser.add_argument('-f', '--force-overwrite', action="store_true",
                        help='Overwrite output file if it does already exist.')
    parser.add_argument('--probability-random-length', default=0.05, required=False, type=float,
                        help="Probability of creating a sample with a random length between 5 and `max_sequence_length`.")
    parser.add_argument('--probability-single-sentence', default=0.1, required=False, type=float,
                        help="Probability of creating a sentence with a single sentence.")
    parser.add_argument('--probability-first-segment-over-length', default=0.5, required=False, type=float,
                        help="Probability of creating a longer first sequence.")

    args = parser.parse_args()

    if args.dataset_names_list:
        logging.info(f"Available datasets {ALL_DATASET_NAMES}")
    else:
        main(args)
