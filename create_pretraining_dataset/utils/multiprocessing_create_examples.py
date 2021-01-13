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

logging.getLogger().setLevel(logging.INFO)


def bert_word_tails(tokenizer, ids):
    return [
        token.startswith('##') for token in tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    ]

def roberta_words_tails(tokenizer, ids):
    return [
        (not token.startswith('Ġ')) and (token not in tokenizer.all_special_tokens) and (i != 1)
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False))
    ]

def gpt_words_tails(tokenizer, ids):
    raise NotImplementedError("This tokenizer is not supported yet")

def xlnet_words_tails(tokenizer, ids):
    raise NotImplementedError("This tokenizer is not supported yet")

def get_words_tails(tokenizer, ids):
    r""" Return words tails built in the right way based on model type. """
    if isinstance(
        tokenizer, (BertTokenizer, BertTokenizerFast, ElectraTokenizer, ElectraTokenizerFast)
    ):
        return bert_word_tails(tokenizer, ids)

    elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
        return roberta_words_tails(tokenizer, ids)

    elif isinstance(tokenizer, (XLNetTokenizer, XLNetTokenizerFast)):
        return xlnet_words_tails(tokenizer, ids)

    elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
        return gpt_words_tails(tokenizer, ids)


class ExampleCreator(object):

    def __init__(
        self,
        tokenizer,
        max_sequence_length,
        do_not_pad,
        add_words_tails,
        probability_single_sentence,
        probability_random_length,
        probability_first_segment_over_length,
    ):
        self.tokenizer = tokenizer
        self.probability_single_sentence = probability_single_sentence
        self.probability_random_length = probability_random_length
        self.probability_first_segment_over_lengt = probability_first_segment_over_length
        
        self.do_not_pad = do_not_pad
        self.add_words_tails = add_words_tails
        self.max_length = max_sequence_length
        self.target_length = max_sequence_length
        self.current_sentences = []
        self.current_lengths = []
    
    def current_length(self):
        return sum(self.current_lengths)

    def add_line(self, line):
        """ Adds a line of text to the current example being built. """
        if (not line) and (self.current_length() > 0):  # empty lines separate docs
            return self.create_example()

        # retrieve line length preview (no special tokens)
        length = self.get_encoded_length(line)

        self.current_sentences.append(line)
        self.current_lengths.append(length)

        if self.current_length() >= self.target_length:
            return self.create_example()
        return None

    def get_encoded_length(self, sentence):
        r""" Get number of expected tokens in this sentence. """
        return len(self.tokenizer.tokenize(sentence))

    def create_example(self):
        r"""
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
                    and random.random() < self.probability_first_segment_over_lengt
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
        if random.random() < self.probability_random_length:
            self.target_length = random.randint(5, self.max_length)
        else:
            self.target_length = self.max_length
        
        return self.make_example(
            first_segment,
            second_segment
        )

    def make_example(
        self,
        first_segment,
        second_segment
    ):
        f""" Converts two "segments" of text into an example. """

        tok_kwargs = {
            'padding': 'do_not_pad' if self.do_not_pad else 'max_length',
            'max_length': self.max_length,
            'verbose': False,
            'truncation': True
        }

        if second_segment:
            example = self.tokenizer(first_segment, second_segment, **tok_kwargs)
        else:
            example = self.tokenizer(first_segment, **tok_kwargs)

        if self.add_words_tails:
            example['words_tails'] = get_words_tails(self.tokenizer, example['input_ids'])

        return dict(example)


def producer(sentences_generator, in_queues, num_processes):
    r""" Fill the input queue with examples taken from the generator and add a counter. """
    i = 0
    for sentence in sentences_generator:
        assert sentence is not None
        in_queues[i].put(sentence)
        if not sentence:
            i = (i + 1) % num_processes

    for j in range(num_processes):
        in_queues[j].put(None)


def worker(
    tokenizer: AutoTokenizer,
    in_queue: Queue,
    out_queue: Queue,
    job_id: int,
    add_words_tails: bool = True,
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
        add_words_tails=add_words_tails,
        probability_single_sentence=probability_single_sentence,
        probability_random_length=probability_random_length,
        probability_first_segment_over_length=probability_first_segment_over_length
    )

    while True:
        sentence = in_queue.get()
        if sentence is None:
            example = examples_builder.add_line("")
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
    sentences: Generator,
    tokenizer: AutoTokenizer,
    max_sequence_length: int = None,
    do_not_pad: bool = False,
    add_words_tails: bool = True,
    probability_random_length: float = 0.05,
    probability_single_sentence: float = 0.1,
    probability_first_segment_over_length: float = 0.5,
    num_processes: int = 1
):
    r""" `sentences` is a generator providing sentences separated in documents by empty sentence. """

    in_queues = [Queue() for _ in range(num_processes)]
    out_queues = [Queue() for _ in range(num_processes)]

    workers = [
        Process(target=worker, args=(tokenizer, in_queues[i], out_queues[i], i),
                kwargs={
                    'max_sequence_length': max_sequence_length,
                    'do_not_pad': do_not_pad,
                    'add_words_tails': add_words_tails,
                    'probability_random_length': probability_random_length,
                    'probability_single_sentence': probability_single_sentence,
                    'probability_first_segment_over_length': probability_first_segment_over_length
                }
        ) for i in range(num_processes)
    ]

    for w in tqdm(workers, total=num_processes, desc="Starting example workers"):
        w.start()

    producer_thread = Thread(target=producer, args=(sentences, in_queues, num_processes))
    producer_thread.start()

    yield from tqdm(consumer(out_queues, num_processes), desc="Creating examples", position=1)

    logging.info("Waiting for processes and threads to finish")
    producer_thread.join()

    for w in workers:
        w.join()
