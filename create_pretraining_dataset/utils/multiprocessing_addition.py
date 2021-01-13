import logging
from typing import Generator
from threading import Thread
from multiprocessing import Queue, Process

from compressed_dictionary import CompressedDictionary
from tqdm import tqdm


def producer(examples_generator, in_queues, num_processes):
    r""" Fill the input queues with examples taken from the generator. """
    i = 0

    for line in examples_generator:
        in_queues[i].put(line)
        i = (i + 1) % num_processes

    for i in range(num_processes):
        in_queues[i].put(None)


def worker(in_queue, out_queue, compression):
    r""" While there are samples add them to the dictionary. """
    while True:
        res = in_queue.get()
        if res is None:
            out_queue.put(None)
            break
        res = CompressedDictionary.__compress__(res, compression=compression)
        out_queue.put(res)


def consumer(out_queues, num_processes):
    r""" Read from out_queue and return elements as a generator. """
    i = 0
    terminated = 0

    while True:
        res = out_queues[i].get()
        i = (i + 1) % num_processes

        if res is None:
            terminated += 1
            if terminated == num_processes:
                break
        else:
            yield res


def multiprocessing_addition(
    cdictionary: CompressedDictionary,
    examples_generator: Generator,
    num_processes: int = 1,
    compression: str = 'xz'
):

    in_queues = [Queue() for _ in range(num_processes)]
    out_queues = [Queue() for _ in range(num_processes)]

    workers = [
        Process(target=worker, args=(in_queues[i], out_queues[i], compression)) for i in range(num_processes)
    ]
    for w in tqdm(workers, total=num_processes, desc="(Addition) Starting workers"):
        w.start()

    producer_thread = Thread(target=producer, args=(examples_generator, in_queues, num_processes))
    producer_thread.start()

    base = len(cdictionary)
    for i, res in tqdm(enumerate(consumer(out_queues, num_processes)), desc="(Addition) Lines added to the dict", position=2):
        cdictionary.__add_already_compresses_value__(base + i, res) 

    logging.info("(Addition) Waiting for processes and threads to finish")
    producer_thread.join()
    for w in workers:
        w.join()
