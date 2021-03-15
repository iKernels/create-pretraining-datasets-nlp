import random
import re
import math
from typing import Any, Union, Dict, List

from argparse import ArgumentParser, Namespace

import transformers
from create_pretraining_dataset.strategies import _Strategy
from create_pretraining_dataset.utils import TailsCreator


class Electra(_Strategy):
    r"""
    ELECTRA introduced a slightly modified way to create examples. In ELECTRA sentences 
    are usually paired with some consecutive amount of text that makes sense. Sometimes, the example contains
    only a single sentence and sometimes it is trimmed to short lengths to introduce some padding.
    """

    def __init__(self, hparams: Namespace, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(hparams, tokenizer)
        self.current_sentences = []
        self.current_lengths = []
        self.target_length = self.hparams.max_sequence_length
        self.tails_creator = TailsCreator(tokenizer)

    @property
    def current_length(self):
        return sum(self.current_lengths)

    def get_words_tails(self, ids):
        r""" Return a tensor where `True` means that the corresponding token is not first in a word. """
        return self.tails_creator.get_words_tails(ids)

    def arguments_check(self):
        assert self.hparams.probability_random_length >= 0 and self.hparams.probability_random_length <= 1, (
            "`--probability_random_length` must be None or a positive integer"
        )

        assert self.hparams.probability_single_sentence >= 0 and self.hparams.probability_single_sentence <= 1, (
            "`--probability_single_sentence` must be None or a positive integer"
        )

        assert self.hparams.probability_first_segment_over_length >= 0 and self.hparams.probability_first_segment_over_length <= 1, (
            "`--probability_first_segment_over_length` must be None or a positive integer"
        )

        assert self.hparams.lines_delimiter is not None and len(self.hparams.lines_delimiter) >= 0, (
            "`--lines_delimiter` must be a non-empty string"
        )

    def add_example_to_dict(self, example, examples_dict):
        r""" Add example to dict. """
        for k, v in example.items():
            if k in examples_dict:
                examples_dict[k].append(v)
            else:
                examples_dict[k] = [v]

    def __call__(self, text: List[Any]) -> List:
        r""" Process a batch of texts. """
        new_examples = []

        # for every doc
        for doc in text:
            # for every paragraph
            for paragraph in re.split(self.hparams.lines_delimiter, doc):

                # continue if empty or too short
                if not paragraph.strip():
                    continue
                elif self.hparams.apply_cleaning and self.filter_out(paragraph):
                    continue

                example = self.add_line(paragraph)
                if example:
                    new_examples.append(example)

            if self.current_length != 0:
                example = self.create_example()
                new_examples.append(example)

        return new_examples

    def filter_out(self, line):
        r""" Filter sentence if not long enough. """
        return len(line) < self.hparams.min_line_length

    def clean(self, line):
        r""" () is remainder after link in it filtered out. """
        return line.strip().replace("\n", " ").replace("()","")

    def get_encoded_length(self, sentence):
        r""" Get number of expected tokens in this sentence. """
        return len(self.tokenizer.tokenize(sentence, verbose=False))

    def add_line(self, line):
        r"""Adds a line of text to the current example being built."""
        line = self.clean(line)

        # retrieve line length preview (no special tokens)
        length = self.get_encoded_length(line)

        self.current_sentences.append(line)
        self.current_lengths.append(length)

        if self.current_length >= self.target_length:
            return self.create_example()
        return None

    def create_example(self):
        r"""
        Creates a pre-training example from the current list of sentences.

        First give a small chance to only have one segment as in classification tasks.
        Then, the sentence goes to the first segment if (1) the first segment is
        empty, (2) the sentence doesn't put the first segment over length or
        (3) 50% of the time when it does put the first segment over length
        """

        if random.random() < self.hparams.probability_single_sentence:
            first_segment_target_length = math.inf
        else:
            first_segment_target_length = (self.target_length - 3) // 2

        first_segment, second_segment = "", ""
        first_segment_length, second_segment_length = 0, 0

        # sentence is a string, sentence_len is the corresponding tokenizer length
        for sentence, sentence_len in zip(self.current_sentences, self.current_lengths):
            if (
                (first_segment_length == 0)
                or (first_segment_length + sentence_len < first_segment_target_length)
                or (
                    second_segment_length == 0
                    and first_segment_length < first_segment_target_length
                    and random.random() < self.hparams.probability_first_segment_over_length
                )
            ):
                first_segment += sentence
                first_segment_length += sentence_len
            else:
                second_segment += sentence
                second_segment_length += sentence_len

            # prepare to start building the next example
            self.current_sentences = []
            self.current_lengths = []

        # small chance for random-length instead of max_length-length example
        if random.random() < self.hparams.probability_random_length:
            self.target_length = random.randint(5, self.hparams.max_sequence_length)
        else:
            self.target_length = self.hparams.max_sequence_length

        return self.make_example(
            first_segment,
            second_segment
        )

    def make_example(self, first_segment, second_segment):
        f""" Converts two "segments" of text into an example. """

        tok_kwargs = {
            'padding': 'do_not_pad' if self.hparams.do_not_pad else 'max_length',
            'max_length': self.hparams.max_sequence_length,
            'verbose': False,
            'truncation': True
        }

        if second_segment:
            example = self.tokenizer(first_segment, second_segment, **tok_kwargs)
        else:
            example = self.tokenizer(first_segment, **tok_kwargs)

        if self.hparams.compute_words_tails:
            example['words_tails'] = self.get_words_tails(example['input_ids'])

        return dict(example)

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        parser.add_argument('--probability_random_length', default=0.05, required=False, type=float,
                        help="Probability of creating a sample with a random length between 5 and `max_sequence_length`.")
        parser.add_argument('--probability_single_sentence', default=0.1, required=False, type=float,
                            help="Probability of creating a sentence with a single sentence.")
        parser.add_argument('--probability_first_segment_over_length', default=0.5, required=False, type=float,
                            help="Probability of creating a longer first sequence.")
        parser.add_argument('--lines_delimiter', default='\n', required=False,
                            help="Split documents into lines on this characted (string)")
        parser.add_argument('--min_line_length', type=int, default=80, required=False,
                            help="Minimum line length to consider (in characters)")
        parser.add_argument('--apply_cleaning', action="store_true",
                            help="Clean dataset lines")
        return parser
