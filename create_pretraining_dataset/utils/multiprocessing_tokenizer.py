import logging
from threading import Thread
from multiprocessing import Queue, Process

import transformers
from transformers import (
    BertTokenizer, BertTokenizerFast, ElectraTokenizer, ElectraTokenizerFast,
    RobertaTokenizer, RobertaTokenizerFast, XLNetTokenizer, XLNetTokenizerFast,
    GPT2Tokenizer, GPT2TokenizerFast
)
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)



def bert_word_tails(tokenizer, ids):
    return [
        token.startswith('##') for token in tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    ]

def roberta_words_tails(tokenizer, ids):
    return [False] * len(ids)

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


def producer(sentence_generator, in_queues):
    r""" Read sentences from `sentence_generator` and put them in the queues. """
    i = 0
    num_processes = len(in_queues)

    for line in sentence_generator:
        in_queues[i].put(line)
        i = (i + 1) % num_processes

    for i in range(num_processes):
        in_queues[i].put(None)

def consumer(out_queues):
    r""" Merge tokenizer data from the different queues into a single generator. """
    i = 0
    terminated = 0
    num_processes = len(out_queues)

    while True: 
        res = out_queues[i].get()
        i = (i + 1) % num_processes

        if res is None:
            terminated += 1
            if terminated == num_processes:
                break
        else:
            yield res

def parse_line(line: str, tokenizer: transformers.PreTrainedTokenizer):
    res = dict()
    input_ids = tokenizer.encode(line, add_special_tokens=False, verbose=False)
    res["input_ids"] = input_ids
    res['length'] = len(input_ids)
    res["words_tails"] = get_words_tails(tokenizer, input_ids)
    return res

def worker(in_queue, out_queue, tokenizer):
    r""" Tokenize a single line at a time, reading from `in_queue` and outputting to `out_queue`. """
    while True:
        line = in_queue.get()
        if line is None:
            out_queue.put(None)
            break
        
        res = parse_line(line, tokenizer)
        out_queue.put(res)


def multiprocessing_tokenizer(sentence_generator, tokenizer, num_processes):

    in_queues = [Queue(maxsize=1000) for _ in range(num_processes)]
    out_queues = [Queue(maxsize=1000) for _ in range(num_processes)]

    workers = [
        Process(target=worker, args=(in_queues[i], out_queues[i], tokenizer)) for i in range(num_processes)
    ]
    for w in tqdm(workers, total=num_processes, desc="Starting workers"):
        w.start()

    producer_thread = Thread(target=producer, args=(sentence_generator, in_queues))
    producer_thread.start()

    for res in tqdm(consumer(out_queues), desc="Tokenized lines", position=1):
        yield res

    logging.info("Waiting for processes and threads to finish")
    producer_thread.join()
    for w in workers:
        w.join()
