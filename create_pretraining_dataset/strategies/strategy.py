from argparse import Namespace, ArgumentParser
from typing import Any, List
from abc import ABC, abstractmethod

import transformers


class _Strategy(ABC):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, hparams: Namespace, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, batch: List[Any]) -> List:
        r""" Receive a batch of data and return processed version. """

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        r""" Add strategy specific parameters to the cmd argument parser. """
        return parser