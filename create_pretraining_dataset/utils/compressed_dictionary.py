import os
import math
import lzma
import bz2
import gzip
from collections.abc import MutableMapping
from struct import pack, unpack

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
        compression: compression algorithm, one between `xz`, `gzip` and `bz2`. Defaults to `bz2`.

    Example:
    >>> d = CompressedDictionary()
    >>> d['0'] = 'this is a string!"
    >>> d.to_file('file.bz2')
    >>> a = CompressedDictionary.from_file('file.bz2')
    >>> a == d
    True
    """

    ALLOWED_COMPRESSIONS = { 'xz': lzma, 'gzip': gzip, 'bz2': bz2 }
    ATTRIBUTES_TO_DUMP = ['compression']
    LINE_LENGTH_BYTES = 4

    def __init__(self, compression: str = 'bz2'):
        if not compression in self.ALLOWED_COMPRESSIONS:
            raise ValueError(
                f"`compression` argument not in allowed values: {self.ALLOWED_COMPRESSIONS}"
            )
        self.compression = compression
        self._content = dict()

    @classmethod
    def write_line(cls, data: bytes, fd):
        r""" Write a line composed of a header (data length) and the corresponding payload. """
        payload_length = len(data)
        payload_length_bytes = cls.int2bytes(payload_length, cls.LINE_LENGTH_BYTES)
        line = payload_length_bytes + data
        fd.write(line)
    
    @classmethod
    def write_key_value_line(cls, key, value, fd):
        r""" Pack together key and value and write as line. """
        header = f"i{len(value)}s"
        data = pack(header, key, value)
        cls.write_line(data, fd)

    @classmethod
    def read_line(cls, fd):
        r""" Read a line composed of a header (data length) and the corresponding payload. """
        bytes_payload_length = fd.read(cls.LINE_LENGTH_BYTES)
        if not bytes_payload_length:
            return None # no more data to read

        payload_length = cls.bytes2int(bytes_payload_length)
        data = fd.read(payload_length)
        return data

    @classmethod
    def read_key_value_line(cls, fd):
        r""" Unpack key and value by reading a line. """
        line = cls.read_line(fd)
        if line is None:
            return None # no more data to read

        header = f"i{len(line) - cls.LINE_LENGTH_BYTES}s"
        key, value = unpack(header, line)
        return (key, value)

    def dump(self, filepath: str):
        r"""
        Dump compressed_dictionary to file.
        Start by collecting the attributes that should be saved and then
        move the whole content of the dictionary to the file, separating
        key and values with a tab and different entries with a new-line.
        This is a safe op because json will escape possible tabs and newlines
        contained in the values of the dictionary.
        """
        with self.ALLOWED_COMPRESSIONS[self.compression].open(filepath, "wb") as fo:
            # write arguments
            specs_to_dump = dict()
            for key in self.ATTRIBUTES_TO_DUMP:
                specs_to_dump[key] = getattr(self, key)

            args = self.str2bytes(json.dumps(specs_to_dump))
            self.write_line(args, fo)

            # write key-value pairs
            for k in self.keys():
                self.write_key_value_line(k, self._content[k], fo)

    @classmethod
    def load(cls, filepath: str, compression: str = 'bz2'):
        r"""
        Create an instance by decompressing a dump from disk. First retrieve the
        object internal parameters from the first line of the compressed file,
        then start filling the internal dictionary without doing compression/decompression
        again.
        """
        assert os.path.isfile(filepath), (
            f"`filepath` {filepath} is not a file"
        )

        res = CompressedDictionary()
        with cls.ALLOWED_COMPRESSIONS[compression].open(filepath, "rb") as fi:
            # read and set arguments
            arguments = json.loads(cls.bytes2str(cls.read_line(fi)))
            for key, value in arguments.items():
                setattr(res, key, value)

            # read key-value pairs
            while True:
                line = cls.read_key_value_line(fi)
                if line is None:
                    break
                key, value = line
                res._content[key] = value

        return res

    @staticmethod
    def int2bytes(integer: int, length: int = None):
        r""" Convert integer to bytes computing correct number of needed bytes. """
        needed_bytes = length if length else max(math.ceil((integer).bit_length() / 8), 1)
        return (integer).to_bytes(needed_bytes, byteorder="little")

    @staticmethod
    def bytes2int(byteslist: bytes):
        r""" Convert integer to bytes. """
        return int.from_bytes(byteslist, byteorder="little")

    @staticmethod
    def str2bytes(s):
        return s.encode('utf-8')

    @staticmethod
    def bytes2str(b):
        return b.decode('utf-8')

    @classmethod
    def __compress__(cls, value, compression: str = 'bz2'):
        value = json.dumps(value)
        value = cls.str2bytes(value)
        value = cls.ALLOWED_COMPRESSIONS[compression].compress(value)
        return value

    @classmethod
    def __decompress__(cls, compressed_value, compression: str = 'bz2'):
        value = cls.ALLOWED_COMPRESSIONS[compression].decompress(compressed_value)
        value = cls.bytes2str(value)
        value = json.loads(value)
        return value

    def __getitem__(self, key: int):
        value = self._content[key]
        value = self.__class__.__decompress__(value, compression=self.compression)
        return value

    def __setitem__(self, key: int, value: Union[Dict, List]):
        value = self.__class__.__compress__(value, compression=self.compression)
        self._content[key] = value

    def __add_already_compresses_value__(self, key: int, value: bytes):
        self._content[key] = value
    
    def __get_without_decompress_value__(self, key: int):
        return self._content[key]

    def __delitem__(self, key: int):
        del self._content[key]

    def __iter__(self):
        return iter(self._content)

    def __len__(self):
        return len(self._content)

    def __eq__(self, o: object):
        return super().__eq__(o) and (self.compression == o.compression)

    def __str__(self):
        return f"<CompressedDictionary object at {hash(self)}>"

    def merge(self, other, shift_keys=True):
        r"""
        Merge another dictionary with this one. If `shift_keys` is True,
        duplicated keys will be shifter in `other` to free positions. Otherwise,
        an error is raised.
        Dictionaries must use the same `compression` algorithm.
        """
        if self.compression != other.compression:
            raise ValueError(
                f"`other` must use the same `compression` algorithm as `self`"
            )

        res = CompressedDictionary()
        for key in self.keys():
            res.__add_already_compresses_value__(key, self.__get_without_decompress_value__(key))
        
        for key in other.keys():
            if key in res:
                if shift_keys:
                    res.__add_already_compresses_value__(len(res), other.__get_without_decompress_value__(key))
                else:
                    raise ValueError(
                        f"There is a common key {key} between `self` and `other`"
                    )
            else:
                res.__add_already_compresses_value__(key, other.__get_without_decompress_value__(key))

        return res

    def import_from_other(self, other, shift_keys=True):
        r"""
        Merge another dictionary into this one. If `shift_keys` is True,
        duplicated keys will be shifter in `other` to free positions. Otherwise,
        an error is raised.
        Dictionaries must use the same `compression` algorithm.
        """

        if self.compression != other.compression:
            raise ValueError(
                f"`other` must use the same `compression` algorithm as `self`"
            )
        
        for key in list(other.keys()):
            if key in self:
                if shift_keys:
                    self.__add_already_compresses_value__(len(self), other.__get_without_decompress_value__(key))
                else:
                    raise ValueError(
                        f"There is a common key {key} between `self` and `other`"
                    )
            else:
                self.__add_already_compresses_value__(key, other.__get_without_decompress_value__(key))



if __name__ == "__main__":

    import random

    def generate_dict(depth=0):
        res = dict()
        if depth > 4:
            return None
        for i in range(random.randint(2, 5)):
            if random.random() > 0.9:
                res[len(res)] = generate_dict(depth=depth+1)
            elif random.random() > 0.45:
                res[len(res)] = generate_list(depth=depth+1)
            elif random.random() > 0.2:
                res[len(res)] = generate_string()
            else:
                res[len(res)] = random.random()
        return res

    def generate_string():
        return ''.join([chr(random.randint(0, 2**20)) for _ in range(50)])

    def generate_list(depth=0):
        res = []
        if depth > 4:
            return None
        for i in range(random.randint(2, 5)):
            if random.random() > 0.9:
                res.append(generate_dict(depth=depth+1))
            elif random.random() > 0.45:
                res.append(generate_list(depth=depth+1))
            elif random.random() > 0.2:
                res.append(generate_string())
            else:
                res.append(random.random())
        return res

    # testing save/reload
    for i in range(1000):
        
        dd = CompressedDictionary()
        for i in range(10):
            if random.random() > 0.7:
                dd[i] = generate_dict()
            elif random.random() > 0.5:
                dd[i] = generate_list()
            elif random.random() > 0.25:
                dd[i] = generate_string()
            else:
                dd[i] = random.random()

        dd.dump("tmp.bz2")
        a = CompressedDictionary.load('tmp.bz2')

        assert a == dd

        dd = dd.merge(dd, shift_keys=True)
        dd.import_from_other(dd, shift_keys=True)
