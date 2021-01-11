import logging
import math
import queue
import random
from multiprocessing import Process, Queue
from threading import Thread
from typing import Generator

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertTokenizer, BertTokenizerFast,
    RobertaTokenizer, RobertaTokenizerFast,
    XLNetTokenizer, XLNetTokenizerFast,
    GPT2Tokenizer, GPT2TokenizerFast,
    ElectraTokenizer, ElectraTokenizerFast
)

DUMMY_LINE = { 'input_ids': [], 'words_tails': [], 'length': 0 }


class SpecialTokensMap:
    r""" Parse a tokenizer and set all the common attrbutes. """

    start_id: int
    separator_id: int
    end_id: int
    pad_id: int
    padding_side: str
    use_token_types: bool

    def __init__(self, tokenizer):

        if isinstance(
            tokenizer, (BertTokenizer, BertTokenizerFast, ElectraTokenizer, ElectraTokenizerFast)
        ):
            self.set_tokenizer_parameters(
                start_id=tokenizer.cls_token_id,
                separator_id=tokenizer.sep_token_id,
                end_id=tokenizer.sep_token_id,
                pad_id=tokenizer.pad_token_id,
                padding_side=tokenizer.padding_side,
                use_token_types=True
            )

        elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.set_tokenizer_parameters(
                start_id=tokenizer.bos_token_id,
                separator_id=tokenizer.sep_token_id,
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                padding_side=tokenizer.padding_side,
                use_token_types=False
            )

        elif isinstance(tokenizer, (XLNetTokenizer, XLNetTokenizerFast)):
            raise NotImplementedError("This tokenizer is not supported yet")

        elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            raise NotImplementedError("This tokenizer is not supported yet")

    def set_tokenizer_parameters(
        self,
        start_id: int,
        separator_id: int,
        end_id: int,
        pad_id: int,
        padding_side: str,
        use_token_types: bool
    ):
        self.start_id = start_id
        self.separator_id = separator_id
        self.end_id = end_id
        self.pad_id = pad_id
        self.padding_side = padding_side
        self.use_token_types = use_token_types


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
        self.special_tokens_map = SpecialTokensMap(tokenizer=tokenizer)
        self.probability_single_sentence = probability_single_sentence
        self.probability_random_length = probability_random_length
        self.probability_first_segment_over_lengt = probability_first_segment_over_length
        
        self.do_not_pad = do_not_pad
        self.max_length = max_sequence_length
        self.target_length = max_sequence_length
        self.current_sentences = []
        self.current_words_tails = []
        self.current_length = 0

    def add_line(self, line):
        """ Adds a line of text to the current example being built. """
        if line['length'] == 0 and self.current_length > 0:  # empty lines separate docs
            return self.create_example()

        self.current_sentences.append(line['input_ids'])
        self.current_words_tails.append(line['words_tails'])
        self.current_length += line['length']

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

        # [start] + sentence
        input_ids = [self.special_tokens_map.start_id] + first_segment
        words_tails = [False] + first_words_tails
        token_type_ids = [0] * len(input_ids)

        if second_segment:
            # possibly add [SEP] + second sentence
            input_ids += [self.special_tokens_map.separator_id] + second_segment
            words_tails += [False] + second_words_tails
            token_type_ids += [1] * (len(second_segment) + 1)

        # add final tokens
        input_ids += [self.special_tokens_map.end_id]
        words_tails += [False]
        token_type_ids += [1]
        
        attention_mask = [1] * len(input_ids)

        # pad either on left or right side
        if not self.do_not_pad:

            # pad either on left or right side
            if self.special_tokens_map.padding_side == 'right':
                input_ids += [self.special_tokens_map.pad_id] * (self.max_length - len(input_ids))
                words_tails += [False] * (self.max_length - len(words_tails))
                attention_mask += [0] * (self.max_length - len(attention_mask))
                token_type_ids += [0] * (self.max_length - len(token_type_ids))
            else:
                input_ids = [self.special_tokens_map.pad_id] * (self.max_length - len(input_ids)) + input_ids
                words_tails = [False] * (self.max_length - len(words_tails)) + words_tails
                attention_mask = [0] * (self.max_length - len(attention_mask)) + attention_mask
                token_type_ids = [0] * (self.max_length - len(token_type_ids)) + token_type_ids

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_tails": words_tails
        }

        if self.special_tokens_map.use_token_types:
            example["token_type_ids"] = token_type_ids

        return example


def producer(examples_generator, in_queues, num_processes):
    r""" Fill the input queue with examples taken from the generator and add a counter. """
    i = 0
    for example in examples_generator:
        assert example is not None
        in_queues[i].put(example)
        if example['length'] == 0:
            i = (i + 1) % num_processes

    for j in range(num_processes):
        in_queues[j].put(None)


def worker(
    tokenizer: AutoTokenizer,
    in_queue: Queue,
    out_queue: Queue,
    job_id: int,
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

    while True:
        sentence = in_queue.get()
        if sentence is None:
            example = examples_builder.add_line(DUMMY_LINE)
            if example is not None:
                out_queue.put(example)
            out_queue.put(None)
            break

        example = examples_builder.add_line(sentence)
        if example is not None:
            out_queue.put(example)

def consumer(out_queues, num_processes):
    r"""
    Read from out_queue and return elements as a generator. 
    We use a timeout because there is the remote possibility, since the input queues are unbalanced,
    that the producer is stuck at filling a queue while the consumer is waiting for something in some
    other output queue. This is especially true when the queues have a max size.
    """
    i = 0
    terminated = [False] * num_processes

    while True:
        if not terminated[i]:
            try:
                res = out_queues[i].get(timeout=1)
                if res is None:
                    terminated[i] = True
                    if all(terminated):
                        break
                else:
                    yield res        
            except queue.Empty:
                pass
        i = (i + 1) % num_processes


def multiprocessing_create_examples(
    tokenized_sentences: Generator,
    tokenizer: AutoTokenizer,
    max_sequence_length: int = None,
    do_not_pad: bool = False,
    probability_random_length: float = 0.05,
    probability_single_sentence: float = 0.1,
    probability_first_segment_over_length: float = 0.5,
    num_processes: int = 1
):

    in_queues = [Queue() for _ in range(num_processes)]
    out_queues = [Queue() for _ in range(num_processes)]

    workers = [
        Process(target=worker, args=(tokenizer, in_queues[i], out_queues[i], i),
                kwargs={
                    'max_sequence_length': max_sequence_length,
                    'do_not_pad': do_not_pad,
                    'probability_random_length': probability_random_length,
                    'probability_single_sentence': probability_single_sentence,
                    'probability_first_segment_over_length': probability_first_segment_over_length
                }
        ) for i in range(num_processes)
    ]

    for w in tqdm(workers, total=num_processes, desc="Starting example workers"):
        w.start()

    producer_thread = Thread(target=producer, args=(tokenized_sentences, in_queues, num_processes))
    producer_thread.start()

    yield from tqdm(consumer(out_queues, num_processes), desc="Creating examples", position=2)

    logging.info("Waiting for processes and threads to finish")
    producer_thread.join()

    for w in workers:
        w.join()
