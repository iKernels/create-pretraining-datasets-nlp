import os
import lzma
import bz2
import gzip
import pickle
from collections.abc import MutableMapping

import json
from typing import Dict, List, Union


class CompressedDictionary(MutableMapping):
    r"""
    A dictionary where every value is compressed. Values can be dictionaries, lists or strings.
    Contains also primitives to be dumped to file and restored from a file.

    This dictionary is multithread-safe and can be easily used with multiple thread calling both get and set.

    Performance:
    - Compression of about 225 entries / second. Tested with values equal to strings with an average length of 2000 characters each.
    - Decompression of about 2000 entries / second. Entries are the one compressed above. 
    
    Args:
        compression: compression algorithm, one between `xz`, `gzip` and `bz2`
        initial_data: dictionary of initial data (entries will be compressed)
    
    Example:
    >>> d = CompressedDictionary()
    >>> d['0'] = 'this is a string!"
    >>> d.to_file('file.xz')
    >>> a = CompressedDictionary.from_file('file.xz')
    >>> a == d
    True
    """

    ALLOWED_COMPRESSIONS = { 'xz': lzma, 'gzip': gzip, 'bz2': bz2 }

    def __init__(self, compression: str = 'xz', initial_data: dict = None):
        if not compression in self.ALLOWED_COMPRESSIONS:
            raise ValueError(
                f"`compression` argument not in allowed values: {self.ALLOWED_COMPRESSIONS}"
            )
        self.compression = compression
        self.content = dict()

        if initial_data is not None:
            self.update(initial_data)

    def asbytes(self):
        bytes_representation = pickle.dumps(self)
        return bytes_representation

    @staticmethod
    def asobject(bytes_representation):
        return pickle.loads(bytes_representation)

    @classmethod
    def from_file(cls, filepath: str, compression: str = 'xz'):
        r"""
        If compression is None, it will try to infer it automagically.
        """
        assert os.path.isfile(filepath), (
            f"`filepath` {filepath} is not a file"
        )

        if compression is None:
            with open(filepath, "rb") as fi:
                return cls.asobject(fi.read())
        else:
            with cls.ALLOWED_COMPRESSIONS[compression].open(filepath, "rb") as fi:
                return cls.asobject(fi.read())

    def to_file(self, filepath: str, compression: str = 'xz'):
        r"""
        Dump compressed_dictionary to file.
        """
        representation = self.asbytes()
        if compression is None:
            with open(filepath, "wb") as fo:
                fo.write(representation)
        else:
            with self.ALLOWED_COMPRESSIONS[self.compression].open(filepath, "wb") as fo:
                fo.write(representation)

    @staticmethod
    def str2bytes(s):
        return s.encode('utf-8')

    @staticmethod
    def bytes2str(b):
        return b.decode('utf-8')

    def __getitem__(self, key: str):
        value = self.content[key]
        value = self.ALLOWED_COMPRESSIONS[self.compression].decompress(value)
        value = self.__class__.bytes2str(value)
        value = json.loads(value)
        return value

    def __setitem__(self, key: str, value: Union[Dict, List]):
        value = json.dumps(value)
        value = self.__class__.str2bytes(value)
        value = self.ALLOWED_COMPRESSIONS[self.compression].compress(value)
        self.content[key] = value

    def __delitem__(self, key: str):
        del self.content[key]

    def __iter__(self):
        return iter(self.content)

    def __len__(self):
        return len(self.content)

    def __eq__(self, o: object):
        return super().__eq__(o) and (self.compression == o.compression)

    def __str__(self):
        return f"<CompressedDictionary object at {hash(self)}>"
    
    def merge(self, other: CompressedDictionary) -> CompressedDictionary:
        raise NotImplementedError()