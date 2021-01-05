import random
import math
from typing import Generator
from tqdm import tqdm

from transformers.models.auto.tokenization_auto import AutoTokenizer


class ExampleCreator(object):

    def __init__(
        self,
        tokenizer,
        max_sequence_length,
        do_not_pad,
        probability_single_sentence,
        probability_random_length,
        probability_first_segment_over_length,
    ):
        self.tokenizer = tokenizer
        self.probability_single_sentence = probability_single_sentence
        self.probability_random_length = probability_random_length
        self.probability_first_segment_over_lengt = probability_first_segment_over_length
        
        self.do_not_pad = do_not_pad
        self.max_length = max_sequence_length
        self.target_length = max_sequence_length
        self.current_sentences = []
        self.current_words_tails = []
        self.current_length = 0

    def remove_special_ids(self, line):
        res = dict()
        res['input_ids'] = [
            x for x in line['input_ids'] if x not in self.tokenizer.all_special_ids
        ]
        res['words_tails'] = [
            t for t, x in zip(line['words_tails'], line['input_ids']) if x not in self.tokenizer.all_special_ids
        ]
        res['length'] = len(res['input_ids'])
        return res

    def add_line(self, line):
        """ Adds a line of text to the current example being built. """
        if line['length'] <= 2 and self.current_length > 0:  # empty lines separate docs
            return self.create_example()

        stripped_line = self.remove_special_ids(line)
        self.current_sentences.append(stripped_line['input_ids'])
        self.current_words_tails.append(stripped_line['words_tails'])
        self.current_length += stripped_line['length']

        if self.current_length >= self.target_length:
            return self.create_example()
        return None

    def create_example(self):
        """
        Creates a pre-training example from the current list of sentences.
        
        First give a small chance to only have one segment as in classification tasks.
        Then, the sentence goes to the first segment if (1) the first segment is
        empty, (2) the sentence doesn't put the first segment over length or
        (3) 50% of the time when it does put the first segment over length
        """

        if random.random() < self.probability_single_sentence:
            first_segment_target_length = math.inf
        else:
            first_segment_target_length = (self.target_length - 3) // 2

        first_segment, first_words_tails = [], []
        second_segment, second_words_tails = [], []

        for sentence, words_tails in zip(self.current_sentences, self.current_words_tails):
            if (
                len(first_segment) == 0
                or len(first_segment) + len(sentence) < first_segment_target_length
                or (
                    len(second_segment) == 0
                    and len(first_segment) < first_segment_target_length
                    and random.random() < self.probability_first_segment_over_lengt
                )
            ):
                first_segment += sentence
                first_words_tails += words_tails
            else:
                second_segment += sentence
                second_words_tails += words_tails

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[: self.max_length - 2]
        first_words_tails = first_words_tails[: self.max_length - 2]
        
        second_segment = second_segment[:max(0, self.max_length - len(first_segment) - 3)]
        second_words_tails = second_words_tails[:max(0, self.max_length - len(first_segment) - 3)]

        # prepare to start building the next example
        self.current_sentences = []
        self.current_words_tails = []
        self.current_length = 0

        # small chance for random-length instead of max_length-length example
        if random.random() < self.probability_random_length:
            self.target_length = random.randint(5, self.max_length)
        else:
            self.target_length = self.max_length
        
        return self.make_example(
            first_segment,
            second_segment,
            first_words_tails,
            second_words_tails
        )

    def make_example(
        self,
        first_segment,
        second_segment,
        first_words_tails,
        second_words_tails
    ):
        f""" Converts two "segments" of text into an example. """

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = [cls_token_id] + first_segment + [sep_token_id]
        words_tails = [False] + first_words_tails + [False]
        token_type_ids = [0] * len(input_ids)

        if second_segment:
            input_ids += second_segment + [sep_token_id]
            words_tails += second_words_tails + [False]
            token_type_ids += [1] * (len(second_segment) + 1)

        attention_mask = [1] * len(input_ids)

        if not self.do_not_pad:
            input_ids += [pad_token_id] * (self.max_length - len(input_ids))
            words_tails += [False] * (self.max_length - len(input_ids))
            attention_mask += [pad_token_id] * (self.max_length - len(attention_mask))
            token_type_ids += [pad_token_id] * (self.max_length - len(token_type_ids))

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "words_tails": words_tails
        }
        return example


def create_examples(
    tokenizer: AutoTokenizer,
    tokenized_sentences: Generator,
    max_sequence_length: int = None,
    do_not_pad: bool = False,
    probability_random_length: float = 0.05,
    probability_single_sentence: float = 0.1,
    probability_first_segment_over_length: float = 0.5
):
    r"""
    Create a dataset given the tokenized sentences and the required constants/probabilities.
    ELECTRA style used.
    """

    examples_builder = ExampleCreator(
        tokenizer,
        max_sequence_length,
        do_not_pad=do_not_pad,
        probability_single_sentence=probability_single_sentence,
        probability_random_length=probability_random_length,
        probability_first_segment_over_length=probability_first_segment_over_length
    )

    n_created = 0
    for sentence in tqdm(tokenized_sentences, desc="Creating examples"):

        example = examples_builder.add_line(sentence)
        if example:
            yield example

    example = examples_builder.add_line({ 'input_ids': [], 'words_tails': [], 'length': 0 })
    if example:
        yield example
