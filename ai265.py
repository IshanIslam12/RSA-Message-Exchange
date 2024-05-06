# ai265.py - Python utilities for Modern Cryptography, LIU Spring 2024
# Copyright 2024 Christopher League <league@contrapunctus.net>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Python utilities for Modern Cryptography, LIU Spring 2024

>>> bytes(xor_zip(b"mC=F:_zYD5!0L", b"%&Q*UsZ.+GMTm"))
b'Hello, world!'

This file contains some experimental code for exploring cryptographic tools and
techniques.  It can be dangerous to "roll your own" crypto, so please do not
rely on this code for actual privacy!

I aim to make the code relatively easy to read and try.  This can be imported as
a module into other Python programs or Jupyter notebooks using statements like
these:

    import ai265
    from ai265 import fastpow, Salsa20

It can also be run as a program invoke the unit tests or provide a rudimentary
command-line interface, using commands like these:

    python ai265.py
    python -m ai265
    python -m ai265 help
    python -m ai265 help rand
    python -m ai265 rand 256

My intention is to use only pure Python for the implementation, but the test
code can rely on some external packages.  At the time of writing, they include
hypothesis[1] and pycryptodome[2].

 1. https://hypothesis.readthedocs.io/en/latest/
 2. https://www.pycryptodome.org/

This module may assume you are using Python 3.10 or later.  Please report any
bugs by email.

"""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Container
from contextlib import contextmanager
from datetime import datetime
from datetime import timezone
from http.client import HTTPMessage
from http.client import parse_headers as parse_http_headers

import argparse
import base64
import doctest
import functools
import hashlib
import io
import itertools
import logging
import math
import operator
import os
import quopri
import re
import secrets
import shutil
import struct
import sys
import time
import typing as t
import unittest

t_wbuf = bytearray | memoryview  # Writable byte buffer
t_buf = bytes | t_wbuf  # Readable byte buffer

####################################################################
###                                           ALPHABETIC CIPHERS ###


def rotate_letter(letter: str, offset: int) -> str:
    """Rotate an alphabetic LETTER by an integer OFFSET.  If the given letter is
    not alphabetic ASCII, return it unchanged.

    >>> rotate_letter("A", 5)
    'F'
    >>> rotate_letter("X", 5)
    'C'

    """
    if ord(letter) >= 0x80 or not letter.isalpha():
        return letter
    start = ord("A" if letter.isupper() else "a")
    k = ord(letter) - start
    k = (k + offset) % 26
    return chr(k + start)


####################################################################
###                                               BITS AND BYTES ###


class FixedWordBase:
    "Operations on fixed-size unsigned integers."

    BITS = 0
    POW = 1
    MAX = 0
    BYTES = 0

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        assert cls.__name__.startswith("Word")
        cls.BITS = int(cls.__name__[4:])
        cls.POW = 2**cls.BITS
        cls.MAX = cls.POW - 1
        cls.BYTES = math.ceil(cls.BITS / 8)

    @classmethod
    def rand(cls):
        "Return a genuine random number in the range of this Word."
        return secrets.randbelow(cls.POW)

    @classmethod
    def assert_range(cls, num: int):
        """Ensure that NUM is in the range of this Word.

        >>> Word16.assert_range(Word16.rand())

        """
        assert 0 <= num < cls.POW, hex(num)

    @classmethod
    def add(cls, fst: int, snd: int) -> int:
        """Add fixed-sized unsigned words with appropriate wrap-around.

        >>> Word8.add(100, 200)
        44

        """
        cls.assert_range(fst)
        cls.assert_range(snd)
        return (fst + snd) & cls.MAX

    @classmethod
    def left_rot(cls, num: int, shift: int) -> int:
        """Bitwise left-rotation of NUM by SHIFT.

        >>> for k in range(7):
        ...    z = Word6.left_rot(5, k)
        ...    print(f'{z:06b} {z:02X}')
        000101 05
        001010 0A
        010100 14
        101000 28
        010001 11
        100010 22
        000101 05

        """
        cls.assert_range(num)
        assert 0 <= shift <= cls.BITS
        num <<= shift
        return (num | num >> cls.BITS) & cls.MAX

    @classmethod
    def split(cls, num: int, target) -> t.Iterator[int]:
        """Iterate through smaller chunks of a word, in little-endian order.
        Only supports target sizes that evenly divide our size.

        >>> Word16.hex_grid(Word64.split(0x58edd8db587e3d9f, Word16))
         3d9f 587e d8db 58ed

        """
        assert cls.BITS % target.BITS == 0
        for _ in range(cls.BITS // target.BITS):
            yield num & target.MAX
            num >>= target.BITS

    @classmethod
    def to_bytes(cls, num: int) -> bytes:
        """Pack a word into bytes, in little-endian order.

        >>> Word32.to_bytes(0xa9bfc6d3)
        b'\\xd3\\xc6\\xbf\\xa9'
        """
        cls.assert_range(num)
        return int_to_bytes(num, cls.BYTES)

    @classmethod
    def join(cls, nums: t.Iterable[int]) -> int:
        "Join NUMS into a larger number, in little-endian order."
        result = 0
        k = 0
        for num in nums:
            cls.assert_range(num)
            result |= num << k
            k += cls.BITS
        return result

    @classmethod
    def from_bytes(cls, buf: bytes) -> int:
        "Unpack a word from bytes, in little-endian order."
        assert len(buf) == cls.BYTES
        return int_from_bytes(buf)

    @classmethod
    def hex_grid(
        cls,
        nums: t.Iterable[int],
        cols: t.Optional[int] = None,
        digits: t.Optional[int] = None,
    ):
        """Neatly display the numbers provided by NUMS, using hexadecimal.  Wrap
        the display into COLS columns.  DIGITS specifies the minimum number of
        hexadecimal digits to print for each number.  The defaults are derived
        from the Word size (aiming for 128 bits per line) but can be overridden.

        >>> Word16.hex_grid(0x9c << i & Word16.MAX for i in range(16))
         009c 0138 0270 04e0 09c0 1380 2700 4e00
         9c00 3800 7000 e000 c000 8000 0000 0000

        """
        if digits is None:
            digits = math.ceil(cls.BITS / 4)
        if cols is None:
            cols = math.ceil(32 / digits)
        i = None
        for i, word in enumerate(nums):
            cls.assert_range(word)
            print(
                f" {word:0{digits}x}", end="\n" if i % cols == cols - 1 else ""
            )
        if i % cols != cols - 1:
            print()


class Word6(FixedWordBase):
    "Operations on 6-bit unsigned integers."


class Word8(FixedWordBase):
    "Operations on 8-bit unsigned integers."


class Word16(FixedWordBase):
    "Operations on 16-bit unsigned integers."


class Word32(FixedWordBase):
    "Operations on 32-bit unsigned integers."


class Word64(FixedWordBase):
    "Operations on 64-bit unsigned integers."


class Word70(FixedWordBase):
    "Operations on 70-bit unsigned integers."


class Word128(FixedWordBase):
    "Operations on 128-bit unsigned integers."


class Word256(FixedWordBase):
    "Operations on 256-bit unsigned integers."


class Word512(FixedWordBase):
    "Operations on 512-bit unsigned integers."


class AbstractGroupoid(ABC, Container):
    """A simple representation of an algebraic Group: a callable binary
    operation, an identity value, and an inverse."""

    identity: t.Any

    @abstractmethod
    def __call__(self, fst, snd):  # pragma: no cover
        ...

    @abstractmethod
    def inverse(self, val):
        "Return the inverse of VAL in this group."


@functools.cache
class XorByteIterGroup(AbstractGroupoid):
    """This represents a group that applies XOR to iterable sequences of bytes,
    producing a new byte stream.  The singleton instance of this class is called
    xor_zip.  Like zip(), when one or the other iterable is exhausted, we stop
    iterating.  This example encodes an ASCII byte string by alternating between
    the bytes 03 and 04.  You can see that the consecutive 'l' characters in
    "Hello" are encoded differently, as "oh".

    >>> import itertools
    >>> bytes(xor_zip(b"Hello World!", itertools.cycle([3,4])))
    b'Kaohl$Tkqhg%'
    >>> key = [secrets.randbelow(256) for _ in range(6)]
    >>> ctxt = bytes(xor_zip(b"secret message", key))
    >>> [chr(b) for b in xor_zip(key, ctxt)]
    ['s', 'e', 'c', 'r', 'e', 't']

    Note that the binary operation itself is careful not to expand the iterable
    arguments any further than needed, and not to attempt expansion multiple
    times. However, to test closure, associativity, and other group laws, we
    need to expand multiple times. So in those cases, we assume the iterable is
    a list of integers or a byte string. Also note that identity is represented
    as a cycle, or potentially infinite sequence! We take extra care not to
    expand that, because it would cause an infinite loop.

    """

    identity = itertools.cycle([0])

    def __call__(
        self, yys: t.Iterable[int], zzs: t.Iterable[int]
    ) -> t.Iterable[int]:
        "Merge two byte streams using XOR, producing a new byte stream."
        return (y ^ z for y, z in zip(yys, zzs))

    def __contains__(self, val):
        "Check whether VAL is an iterable sequence of bytes."
        if val is self.identity:
            return True
        try:
            bytes(val)
            return True
        except (TypeError, ValueError):
            return False

    def inverse(self, val: t.Iterable[int]):
        return val


xor_zip = XorByteIterGroup()


def int_from_bytes(buf: t_buf) -> int:
    "Unpack an integer from bytes, in little-endian order."
    return int.from_bytes(buf, byteorder="little")


def int_to_bytes(val: int, nbytes: t.Optional[int] = None) -> bytes:
    "Pack an integer into bytes, in little-endian order."
    if nbytes is None:
        nbytes = math.ceil(val.bit_length() / 8)
    return val.to_bytes(nbytes, byteorder="little")


def xor_bytes_with_int(buf: t_buf, val: int) -> bytes:
    """Merge a byte string with a big integer, convert back to bytes."""
    return int_to_bytes(int_from_bytes(buf) ^ val, len(buf))


####################################################################
###                                                       HASHES ###


def hash_based_auth_code(
    key: bytes, message: bytes, hash_name: str = "sha1"
) -> bytes:
    "Implement the HMAC algorithm. Should be equivalent to hmac.new."
    hash_func = hashlib.new(hash_name)
    inner_pad = int_from_bytes(b"\x36" * hash_func.block_size)
    outer_pad = int_from_bytes(b"\x5c" * hash_func.block_size)
    if len(key) > hash_func.block_size:
        key = hashlib.new(hash_name, key).digest()
    int_key = int_from_bytes(key)
    inner_key = int_to_bytes(int_key ^ inner_pad, hash_func.block_size)
    outer_key = int_to_bytes(int_key ^ outer_pad, hash_func.block_size)
    inner_hash = hashlib.new(hash_name, inner_key + message).digest()
    return hashlib.new(hash_name, outer_key + inner_hash).digest()


####################################################################
###                                            POWERS AND PRIMES ###


def modulo(mod: int):
    """The standard mod operator, but with curried (and flipped) arguments.

    >>> hun = modulo(100)
    >>> hun(2**8)
    56
    """
    return lambda x: x % mod


def fastpow(base: int, exp: int, mod=lambda x: x, mul=operator.mul) -> int:
    """Fast exponentiation, with a customizable multiplication operator.

    >>> fastpow(3, 10)
    59049
    >>> fastpow(7, 12, mod = lambda x: x % 599)
    125
    >>> fastpow(2, 20, modulo(1000))
    576
    >>> [fastpow(2, k, modulo(10)) for k in range(10)]
    [1, 2, 4, 8, 6, 2, 4, 8, 6, 2]
    """
    assert exp >= 0
    result = mod(1)
    while exp > 0:
        if exp & 1:  # Odd: multiply and decrement
            result = mod(mul(result, base))
            exp -= 1
        else:  # Even: square and halve
            base = mod(mul(base, base))
            exp //= 2
    return result


def bin_mul(fst: int, snd: int, add=operator.add, mod=lambda x: x) -> int:
    """The 'peasant' binary multiplication algorithm, with customizable add and
    optional modulus operators."""
    assert fst >= 0
    result = 0
    while fst > 0:
        if fst & 1:
            result = mod(add(result, snd))
        fst = fst >> 1
        snd = mod(snd << 1)
    return result


def xor_mod(mod: int):
    """Perform a type of modulo by 'subtracting' multiples of MOD from VAL using
    XOR."""

    def xor_mod_loop(val: int) -> int:
        mbits = mod.bit_length()
        vbits = val.bit_length()
        while vbits >= mbits:
            factor = mod << (vbits - mbits)
            val ^= factor
            vbits = val.bit_length()
        return val

    return xor_mod_loop


def byte_mul(
    top: bytes, bot: bytes, mul=operator.mul, add=operator.add, mod=lambda x: x
):
    """Customizable byte-by-byte multiplication."""
    return mod(
        functools.reduce(
            add,
            (
                mod(
                    functools.reduce(
                        add,
                        (
                            mul(bval, tval) << 8 * tidx
                            for tidx, tval in enumerate(top)
                        ),
                    )
                    << 8 * bidx
                )
                for bidx, bval in enumerate(bot)
                if bval != 0
            ),
        )
    )


def crack_discrete_log(
    base: int, mod: int, goal: int, progress: int = Word16.POW
) -> int | None:
    """Search for an integer k such that (base**k)%mod == goal.  Each PROGRESS
    iterations, print a status message, then print when found or not found.

    >>> crack_discrete_log(7, 23, 12)
    crack_discrete_log: found 0x8
    8

    >>> crack_discrete_log(2, 164987, 80662)
    crack_discrete_log: check 0x10000...
    crack_discrete_log: found 0x1fa61
    129633

    >>> crack_discrete_log(2, 10, 5)
    crack_discrete_log: not found

    """
    assert 0 < goal < mod
    result = 1
    for k in range(mod):
        if goal == result:
            print(f"crack_discrete_log: found 0x{k:x}")
            return k
        if k % progress == 0 and k > 0:
            print(f"crack_discrete_log: check 0x{k:x}...")
        result = result * base % mod
    print("crack_discrete_log: not found")
    return None


####################################################################
###                                        SALSA20 STREAM CIPHER ###


class Salsa20(io.RawIOBase):
    """This is the Salsa20 stream cipher.  In this doctest, we will test it
    against the one in PyCryptodome.  Our implementation takes integer inputs,
    where PyCryptodome uses byte strings.
    https://pycryptodome.readthedocs.io/en/latest/src/introduction.html

    >>> key = Salsa20.KEY.rand()
    >>> nonce = Salsa20.NONCE.rand()
    >>> with Salsa20(key, nonce) as sbr:
    ...     our_bytes = sbr.read(672)

    >>> import Crypto.Cipher.Salsa20
    >>> their_bytes = Crypto.Cipher.Salsa20.new(
    ...     Salsa20.KEY.to_bytes(key),
    ...     Salsa20.NONCE.to_bytes(nonce)
    ... ).encrypt(bytes(672))

    >>> our_bytes == their_bytes
    True

    This reader should refuse to read until EOF.

    >>> Salsa20(key).read()
    Traceback (most recent call last):
    NotImplementedError

    """

    KEY = Word256
    NONCE = Word64
    CONST = Word128.from_bytes(b"expand 32-byte k")
    BUFFER_SIZE = 64

    log = logging.getLogger("Salsa20")

    def __init__(self, key: int, nonce: t.Optional[int] = None, block_num=0):
        super().__init__()
        Word64.assert_range(block_num)
        if nonce is None:
            nonce = self.NONCE.rand()
        else:
            self.NONCE.assert_range(nonce)
        #  0:CONST  1:KEY    2:KEY    3:KEY     ivec layout
        #  4:KEY    5:CONST  6:NONCE  7:NONCE
        #  8:BLOCK  9:BLOCK 10:CONST 11:KEY
        # 12:KEY   13:KEY   14:KEY   15:CONST
        key_lo, key_hi = Word256.split(key, Word128)
        self._ivec = [-1] * 16
        self._ivec[0::5] = Word128.split(self.CONST, Word32)
        self._ivec[1:5] = Word128.split(key_lo, Word32)
        self._ivec[6:8] = Word64.split(nonce, Word32)
        self._ivec[11:15] = Word128.split(key_hi, Word32)
        self._offset = 0
        self._calc_block(block_num)
        self._assert_ok()

    def get_nonce(self):
        """Retrieve the nonce value used with this stream.

        >>> sr = Salsa20(0xcafe, nonce=0x7b05443f7d8449f1)
        >>> hex(sr.get_nonce())
        '0x7b05443f7d8449f1'

        """
        return Word32.join(self._ivec[6:8])

    def _assert_ok(self):
        Word6.assert_range(self._offset)
        assert self._buf.bit_length() <= 8 * (self.BUFFER_SIZE - self._offset)

    @staticmethod
    def seekable():
        return True

    def tell(self):
        self._assert_ok()
        return self._get_block_num() << Word6.BITS | self._offset

    def seek(self, pos, whence: int = os.SEEK_SET):
        self._assert_ok()
        if whence == os.SEEK_CUR:
            pos += self.tell()
            assert pos >= 0
            pos &= Word70.MAX
        else:
            assert whence == os.SEEK_SET
            Word70.assert_range(pos)
        new_block_num = pos >> Word6.BITS
        new_offset = pos & Word6.MAX
        if new_block_num != self._get_block_num() or new_offset < self._offset:
            self._calc_block(new_block_num)
        if new_offset > self._offset:
            skip = new_offset - self._offset
            self._buf >>= 8 * skip
            self._offset = new_offset
            self.log.debug("advancing offset by %d bytes", skip)
        return self.tell()

    @staticmethod
    def readable():
        return True

    def read_as_int(self, nbytes: int) -> int:
        """Read up to NBYTES and return them as an integer.  The first element
        of the returned pair is a count of how many bytes actually read (maximum
        64)."""
        self._assert_ok()
        self.log.debug("read_as_int(%d)", nbytes)
        result = 0
        shift = 0
        while nbytes > 0:
            batch_bytes = min(nbytes, self.BUFFER_SIZE - self._offset)
            batch_bits = 8 * batch_bytes
            result |= (self._buf & ((1 << batch_bits) - 1)) << shift
            self.seek(batch_bytes, os.SEEK_CUR)
            nbytes -= batch_bytes
            shift += batch_bits
        return result

    def xor_with_bytes(self, buf: t_buf) -> bytes:
        "XOR the given buffer with the keystream."
        return xor_bytes_with_int(buf, self.read_as_int(len(buf)))

    def readinto(self, buf):
        """
        >>> sr = Salsa20(key = 0, nonce = Word64.MAX)
        >>> Word8.hex_grid(sr.read(64))
         21 f9 f6 4a 7a ba 30 f7 e0 3c 66 38 93 6d 37 5f
         75 3b 68 85 e1 0c 8d d8 91 15 50 26 6c 4b 14 64
         fd 46 cb 6d 93 8f 17 58 30 7a 57 06 d7 27 dc cf
         9e e0 bb 68 a1 03 b4 17 27 1f aa 38 54 a5 52 bf
        >>> sr.seek(16, os.SEEK_CUR)
        80
        >>> Word8.hex_grid(sr.read(80))
         00 41 9a b4 2b 60 f9 28 9f 34 42 61 d0 63 06 e2
         17 e1 e1 12 8f bc 61 6a fd 48 52 00 4a ad fe 2e
         13 5a 77 77 d5 99 7b d1 5b 3f 77 eb f1 c3 5a 97
         ec 84 2e ab 02 78 ab c2 60 44 09 62 50 88 6d 0f
         ab 9f e2 fe 24 ba 25 22 de f5 ee 56 4a ba 4d cf
        """
        nbytes = len(buf)
        buf[:] = int_to_bytes(self.read_as_int(nbytes), nbytes)
        return nbytes

    def readall(self):
        raise NotImplementedError

    def _get_block_num(self):
        return Word32.join(self._ivec[8:10])

    def _calc_block(self, block_num):
        """
        The Aumasson book has two examples of the expected output of shuffling,
        in Listing 5-4.  Here is the expectation with a zero key and position,
        and the maximal nonce.

        >>> sr = Salsa20(key = 0, nonce = Salsa20.NONCE.MAX)
        >>> Word32.hex_grid(sr._ovec)
         e98680bc f730ba7a 38663ce0 5f376d93
         85683b75 a56ca873 26501592 64144b6d
         6dcb46fd 58178f93 8cf54cfe cfdc27d7
         68bbe09e 17b403a1 38aa1f27 54323fe0
        """
        self.log.debug("calculating block number %016x", block_num)
        self._ivec[8:10] = Word64.split(block_num, Word32)
        ovec = self._ivec[:]
        for _ in range(10):
            # Column round
            self._qround(ovec, 0, 4, 8, 12)
            self._qround(ovec, 5, 9, 13, 1)
            self._qround(ovec, 10, 14, 2, 6)
            self._qround(ovec, 15, 3, 7, 11)
            # Row round
            self._qround(ovec, 0, 1, 2, 3)
            self._qround(ovec, 5, 6, 7, 4)
            self._qround(ovec, 10, 11, 8, 9)
            self._qround(ovec, 15, 12, 13, 14)
        self._buf = Word32.join(
            Word32.add(iw, ow) for iw, ow in zip(self._ivec, ovec)
        )
        self._ovec = ovec
        self._offset = 0

    @staticmethod
    def _qround(grid, aaa, bbb, ccc, ddd):
        grid[bbb] ^= Word32.left_rot(Word32.add(grid[aaa], grid[ddd]), 7)
        grid[ccc] ^= Word32.left_rot(Word32.add(grid[bbb], grid[aaa]), 9)
        grid[ddd] ^= Word32.left_rot(Word32.add(grid[ccc], grid[bbb]), 13)
        grid[aaa] ^= Word32.left_rot(Word32.add(grid[ddd], grid[ccc]), 18)

    class Writer(io.RawIOBase):
        "Output stream that enciphers using a Salsa20 keystream."

        def __init__(self, out, key: int, nonce: t.Optional[int] = None):
            super().__init__()
            self._out = out
            self._keystream = Salsa20(key, nonce)
            self._out.write(Salsa20.NONCE.to_bytes(self._keystream.get_nonce()))

        def write(self, buf):
            return self._out.write(self._keystream.xor_with_bytes(buf))

        @staticmethod
        def writable():
            return True

    class Reader(io.RawIOBase):
        "Input stream that deciphers using a Salsa20 keystream."

        def __init__(self, inp, key: int):
            super().__init__()
            self._inp = inp
            nonce = self._inp.read(8)
            assert len(nonce) == 8
            nonce = Salsa20.NONCE.from_bytes(nonce)
            self._keystream = Salsa20(key, nonce)

        def readinto(self, dest):
            nbytes = self._inp.readinto(dest)
            dest[:nbytes] = self._keystream.xor_with_bytes(dest[:nbytes])
            return nbytes

        @staticmethod
        def readable():
            return True

    @staticmethod
    def encipher_bytes(buf: t_buf, key: int, nonce=None) -> bytes:
        """Encipher a byte buffer.  A random nonce will be embedded in result.

        >>> key = Salsa20.KEY.rand()
        >>> c = Salsa20.encipher_bytes(b'Silly', key)
        >>> assert b'Silly' not in c
        >>> Salsa20.decipher_bytes(c, key)
        b'Silly'
        >>> assert b'Silly' not in Salsa20.decipher_bytes(c, key+1)

        """
        out = io.BytesIO()
        Salsa20.encipher_copy_stream(io.BytesIO(buf), out, key, nonce)
        return out.getvalue()

    @staticmethod
    def decipher_bytes(buf: t_buf, key: int) -> bytes:
        "Decipher a byte buffer with embedded nonce."
        out = io.BytesIO()
        Salsa20.decipher_copy_stream(io.BytesIO(buf), out, key)
        return out.getvalue()

    @staticmethod
    def encipher_copy_stream(src, dest, key: int, nonce=None):
        "Copy bytes from SRC to DEST, enciphering using KEY."
        shutil.copyfileobj(src, Salsa20.Writer(dest, key, nonce), 8192)

    @staticmethod
    def decipher_copy_stream(src, dest, key: int):
        "Copy bytes from INP to OUT, deciphering using KEY."
        shutil.copyfileobj(Salsa20.Reader(src, key), dest, 8192)


####################################################################
###                  work in progress: AES/Rijndael BLOCK CIPHER ###


class AES:
    "Pure-python implementation of the Advanced Encryption Standard."

    log = logging.getLogger("AES")
    forward_sbox = bytearray(256)
    inverse_sbox = bytearray(256)
    BLOCK_SIZE = 16

    @staticmethod
    def mul(fst: int, snd: int) -> int:
        "Multiplication in Galois(2**8), with Rijndael polynomial modulus."
        return bin_mul(fst, snd, operator.xor, xor_mod(0x11B))

    @staticmethod
    def round_constant(num: int) -> int:
        """Used in key expansion, round constants are 2**(N-1) in GF(2**8).

        >>> Word8.hex_grid(AES.round_constant(k) for k in range(1,11))
         01 02 04 08 10 20 40 80 1b 36

        """
        assert 1 <= num <= 10
        return fastpow(2, num - 1, mul=AES.mul)

    @staticmethod
    def assert_key_bytes(nbytes: int):
        "Ensure that a key of size NBYTES is compatible with AES."
        assert nbytes in [16, 24, 32]

    @staticmethod
    def sbox_transform(val: int) -> int:
        "The affine transformation applied to inverses to create substitutions."
        return (
            val
            ^ Word8.left_rot(val, 1)
            ^ Word8.left_rot(val, 2)
            ^ Word8.left_rot(val, 3)
            ^ Word8.left_rot(val, 4)
            ^ 0x63
        )

    @classmethod
    def init_substitution_boxes(cls):
        "Initialize the Rijndael substitution boxes."
        cls.forward_sbox[0] = 0x63
        cls.inverse_sbox[0x63] = 0
        ppp = qqq = 1
        for _ in range(255):
            # The bytes 03 and F6 are multiplicative inverses, but also they
            # are generators for the entire field.
            ppp = AES.mul(3, ppp)
            qqq = AES.mul(0xF6, qqq)
            val = cls.sbox_transform(qqq)
            cls.forward_sbox[ppp] = val
            cls.inverse_sbox[val] = ppp
        cls.log.debug("FORWARD S-BOX:")
        cls.show_sbox(cls.forward_sbox, prn=cls.log.debug)
        cls.log.debug("INVERSE S-BOX:")
        cls.show_sbox(cls.inverse_sbox, prn=cls.log.debug)

    @staticmethod
    def show_sbox(box, prn=print):
        "Print out a substitution box with row/column headings."
        assert len(box) == 256
        prn(" " * 5 + " ".join(f"{col:2X}" for col in range(16)))
        for row in range(16):
            prn(
                f"  {row:X}: "
                + " ".join(f"{box[row*16+col]:02x}" for col in range(16))
            )

    def __init__(self, key: t_buf):
        """Constructor for a particular key. Performs immediate expansion.

        Following test is the example from these lecture notes
        https://www.kavaliro.com/wp-content/uploads/2014/03/AES.pdf

        >>> AES(b"Thats my Kung Fu").show_round_keys()
        Round key  0: 54686174 73206d79 204b756e 67204675
        Round key  1: e232fcf1 91129188 b159e4e6 d679a293
        Round key  2: 56082007 c71ab18f 76435569 a03af7fa
        Round key  3: d2600de7 157abc68 6339e901 c3031efb
        Round key  4: a11202c9 b468bea1 d75157a0 1452495b
        Round key  5: b1293b33 05418592 d210d232 c6429b69
        Round key  6: bd3dc287 b87c4715 6a6c9527 ac2e0e4e
        Round key  7: cc96ed16 74eaaa03 1e863f24 b2a8316a
        Round key  8: 8e51ef21 fabb4522 e43d7a06 56954b6c
        Round key  9: bfe2bf90 4559fab2 a16480b4 f7f1cbd8
        Round key 10: 28fddef8 6da4244a ccc0a4fe 3b316f26
        """
        self.assert_key_bytes(len(key))
        self._num_rounds = {16: 10, 24: 12, 32: 14}[len(key)]
        # Key expansion schedule
        key_words = memoryview(key).cast("I")
        kex_buf = memoryview(bytearray(16 * (self._num_rounds + 1)))
        kex_words = kex_buf.cast("I")
        for i, _ in enumerate(kex_words):
            kex_bytes = kex_buf[4 * i :]
            if i < len(key_words):
                kex_words[i] = key_words[i]
            elif i % len(key_words) == 0:
                kex_words[i] = kex_words[i - 1]
                self.rot_word(kex_bytes)
                self.sub_bytes(kex_bytes, self.forward_sbox)
                kex_words[i] ^= kex_words[i - len(key_words)]
                kex_words[i] ^= self.round_constant(i // len(key_words))
            elif len(key_words) > 6 and i % len(key_words) == 4:
                kex_words[i] = kex_words[i - 1]
                self.sub_bytes(kex_bytes, self.forward_sbox)
                kex_words[i] ^= kex_words[i - len(key_words)]
            else:
                kex_words[i] = kex_words[i - 1]
                kex_words[i] ^= kex_words[i - len(key_words)]
        self._round_keys = [k[0] for k in struct.iter_unpack("16s", kex_buf)]
        self._round_keys_long = [k[0] for k in struct.iter_unpack("L", kex_buf)]
        self.show_round_keys(self.log.debug)

    def show_round_keys(self, prn=print):
        "Print out the list of round keys."
        for rnum, rkey in enumerate(self._round_keys):
            prn(f"Round key {rnum:2}: {rkey.hex(' ',4)}")

    @staticmethod
    def rot_word(word: t_wbuf, nbytes: int = 1):
        """Rotate a 4-byte buffer word in-place.

        >>> bs = bytearray(b'FARM.-!')
        >>> AES.rot_word(bs); bs
        bytearray(b'ARMF.-!')
        >>> AES.rot_word(bs, 2); bs
        bytearray(b'MFAR.-!')
        >>> AES.rot_word(bs, 3); bs
        bytearray(b'RMFA.-!')
        """
        assert 0 < nbytes < 4
        front = bytes(word[:nbytes])
        back = bytes(word[nbytes:4])
        word[: 4 - nbytes], word[4 - nbytes : 4] = back, front

    @staticmethod
    def sub_bytes(vec: t_wbuf, sbox: t_buf):
        """In-place substitution of each byte, using sbox.

        >>> bs = bytearray.fromhex("3f1a2b00")
        >>> AES.sub_bytes(bs, AES.forward_sbox); bs.hex()
        '75a2f163'
        >>> AES.sub_bytes(bs, AES.inverse_sbox); bs.hex()
        '3f1a2b00'
        """
        for idx, byte in enumerate(vec):
            vec[idx] = sbox[byte]

    @staticmethod
    def poly_mul(fst: t_buf, snd: t_buf) -> bytes:
        "Polynomial multiplication in GF(2**8)."
        assert len(fst) == len(snd) == 4
        return Word32.to_bytes(
            byte_mul(
                fst,
                snd,
                mul=AES.mul,
                add=operator.xor,
                mod=xor_mod(0x100000001),
            )
        )

    @staticmethod
    def mix_column(vec: t_wbuf):
        """The Rijndael MixColumns multiplication.

        Test vectors are from https://en.wikipedia.org/wiki/Rijndael_MixColumns

        >>> vec = bytearray.fromhex("db135345")
        >>> AES.mix_column(vec); vec.hex()
        '8e4da1bc'
        >>> vec = bytearray.fromhex("f20a225c")
        >>> AES.mix_column(vec); vec.hex()
        '9fdc589d'
        >>> vec = bytearray.fromhex("01010101")
        >>> AES.mix_column(vec); vec.hex()
        '01010101'

        """
        vec[:] = AES.poly_mul(vec, b"\x02\x01\x01\x03")

    @staticmethod
    def unmix_column(vec):
        """Inverse Rijndael MixColumns.

        >>> vec = bytearray.fromhex("c6c6c6c6")
        >>> AES.unmix_column(vec); vec.hex()
        'c6c6c6c6'
        >>> vec = bytearray.fromhex("d5d5d7d6")
        >>> AES.unmix_column(vec); vec.hex()
        'd4d4d4d5'
        >>> vec = bytearray.fromhex("4d7ebdf8")
        >>> AES.unmix_column(vec); vec.hex()
        '2d26314c'
        """
        assert len(vec) == 4
        vec[:] = AES.poly_mul(vec, b"\x0e\x09\x0d\x0b")

    @classmethod
    def key_gen(cls, nbytes=16) -> bytes:
        "Create a uniform-random key compatible with AES."
        cls.assert_key_bytes(nbytes)
        return secrets.token_bytes(nbytes)

    @staticmethod
    def show_block(
        block: bytes | bytearray | memoryview, heading="", prn=print
    ):
        "Print 16 bytes as a matrix, assuming column-major order."
        for row_idx in range(4):
            row = block[row_idx:16:4]
            out = io.StringIO()
            out.write(" [" if row_idx == 0 else "  ")
            out.write(row.hex(" "))
            out.write("]  |" if row_idx == 3 else "   |")
            for byte in row:
                out.write(chr(byte) if 0x20 <= byte < 0x7F else ".")
            out.write("|")
            if row_idx == 0 and len(heading) > 0:
                out.write("  # ")
                out.write(heading)
            prn(out.getvalue())
            out.seek(0)
            out.truncate()

    def enc_block(self, buf):

        """Encrypt one 16-byte block using the key.

        Example from https://www.kavaliro.com/wp-content/uploads/2014/03/AES.pdf

        >>> AES(b"Thats my Kung Fu").enc_block(b"Two One Nine Two").hex()
        '29c3505f571420f6402299b31a02d73a'

        Example from Aumasson book:

        >>> aes = AES(bytes.fromhex("2c6202f9a582668aa96d511862d8a279"))
        >>> aes.enc_block(bytes([0] * 16)).hex()
        '12b620bb5eddcde9a07523e59292a6d7'

        """
        assert len(buf) == self.BLOCK_SIZE
        block = memoryview(bytearray(buf))  # Ensure block is mutable
        # These are then mutable slices or views of the block:
        block_longs = block.cast("L")  # as array of 2 longs (64 bits each)
        block_cols = [block[i : i + 4] for i in range(0, 16, 4)]
        block_rows = [block[r::4] for r in range(4)]
        self.show_block(block, "Initial state", self.log.debug)

        for round_num, round_key in enumerate(self._round_keys):
            # Steps: SubBytes, ShiftRows, MixColumns, AddRoundKey.
            # First round (round_num 0) does AddRoundKey only.
            if 0 < round_num:
                self.sub_bytes(block, self.forward_sbox)
                self.show_block(block, f"SubBytes {round_num}", self.log.debug)

                for row_idx, row_slice in enumerate(block_rows[1:], start=1):
                    self.rot_word(row_slice, row_idx)
                self.show_block(block, f"ShiftRows {round_num}", self.log.debug)

            # Last round skips MixColumns.
            if 0 < round_num < self._num_rounds:
                for col in block_cols:
                    self.mix_column(col)
                self.show_block(
                    block, f"MixColumns {round_num}", self.log.debug
                )

            # AddRoundKey: remember that '^=' means "XOR with"
            self.show_block(round_key, f"Key {round_num}", self.log.debug)
            block_longs[0] ^= self._round_keys_long[2 * round_num]  # least
            block_longs[1] ^= self._round_keys_long[2 * round_num + 1]
            self.show_block(block, f"AddRoundKey {round_num}", self.log.debug)

        return bytes(block)

    def dec_block(self, buf):
        """Decrypt one 16-byte block using the key.

        >>> ciphertext = bytes.fromhex("29c3505f571420f6402299b31a02d73a")
        >>> AES(b"Thats my Kung Fu").dec_block(ciphertext)
        b'Two One Nine Two'
        """
        assert len(buf) == self.BLOCK_SIZE
        block = memoryview(bytearray(buf))  # Ensure block is mutable
        # These are then mutable slices or views of the block:
        block_longs = block.cast("L")  # as array of 2 longs (64 bits each)
        block_cols = [block[i : i + 4] for i in range(0, 16, 4)]
        block_rows = [block[r::4] for r in range(4)]
        self.show_block(block, "Initial state", self.log.debug)

        for round_num in range(len(self._round_keys) - 1, -1, -1):
            round_key = self._round_keys[round_num]
            # Steps: SubBytes, ShiftRows, MixColumns, AddRoundKey.
            # AddRoundKey: remember that '^=' means "XOR with"
            self.show_block(round_key, f"Key {round_num}", self.log.debug)
            block_longs[0] ^= self._round_keys_long[2 * round_num]  # least
            block_longs[1] ^= self._round_keys_long[2 * round_num + 1]
            self.show_block(block, f"AddRoundKey {round_num}", self.log.debug)

            # Last round skips MixColumns.
            if 0 < round_num < self._num_rounds:
                for col in block_cols:
                    self.unmix_column(col)
                self.show_block(
                    block, f"MixColumns {round_num}", self.log.debug
                )

            # First round (round_num 0) does AddRoundKey only.
            if 0 < round_num:
                for row_idx, row_slice in enumerate(block_rows[1:], start=1):
                    self.rot_word(row_slice, 4 - row_idx)
                self.show_block(block, f"ShiftRows {round_num}", self.log.debug)

                self.sub_bytes(block, self.inverse_sbox)
                self.show_block(block, f"SubBytes {round_num}", self.log.debug)

        return bytes(block)


AES.init_substitution_boxes()


####################################################################
###                                ASCII AND BINARY FILE FORMATS ###


class FileFormats:
    "Readers and writers for keys, parameters, and ciphertexts."
    MAGIC = b":!C:"
    DASHES = "-" * 5

    # Banners for different file types
    KEY = "KEY"
    S20_DATA = "SALSA20 DATA"
    QUO_PRI = "QUOTED PRINTABLE TEXT"

    @staticmethod
    def require_read(inp: io.RawIOBase, nbytes: int) -> bytes:
        "Read NBYTES from INP stream, or face an AssertionError."
        buf = inp.read(nbytes)
        assert buf is not None and len(buf) == nbytes
        return buf

    @staticmethod
    def write_bytes(buf: bytes, out: io.RawIOBase):
        """Write BUF to OUT, prefixed by its size in bytes.  The size is written
        as two bytes, so this limits buffer size to 64 KiB."""
        assert out.writable()
        out.write(Word16.to_bytes(len(buf)))
        out.write(buf)

    @classmethod
    def read_bytes(cls, inp: io.RawIOBase) -> bytes:
        "Read a sized byte buffer from INP."
        assert inp.readable()
        nbytes = Word16.from_bytes(cls.require_read(inp, 2))
        return cls.require_read(inp, nbytes)

    @classmethod
    def write_int(cls, val: int, out: io.RawIOBase):
        """Write VAL to OUT, prefix by its size in bytes.

        >>> out = io.BytesIO()
        >>> FileFormats.write_int(0x7fade, out)
        >>> out.getvalue().hex()
        '0300defa07'
        """
        cls.write_bytes(int_to_bytes(val), out)

    @classmethod
    def read_int(cls, inp: io.RawIOBase):
        """Read VAL from INP.

        >>> hex(FileFormats.read_int(io.BytesIO(bytes.fromhex("0200FECA"))))
        '0xcafe'
        """
        return int_from_bytes(cls.read_bytes(inp))

    t_source = str | bytes | io.BytesIO | io.TextIOWrapper

    @staticmethod
    def wrapped_io(src: t_source = b"") -> io.TextIOWrapper:
        """Create a stream suitable for reading or writing our formats, using
        SRC as a starting point.  SRC may be a character or byte string, or a
        BytesIO.  If it's already a TextIOWrapper, just return it."""
        if isinstance(src, str):
            src = src.encode("ascii")
        if isinstance(src, bytes):
            src = io.BytesIO(src)
        if isinstance(src, io.BytesIO):
            src = io.TextIOWrapper(src, encoding="ascii")
        return src

    class BinWriter(io.RawIOBase):
        """A light wrapper that prefaces binary output with a format-identifying
        magic line and clear-text ASCII headers.  Unlike BufferedWriter, we do
        NOT propagate close() to the underlying stream."""

        def __init__(self, out, banner: str, headers: HTTPMessage):
            super().__init__()
            assert out.writable()
            self._out = out
            out.write(FileFormats.MAGIC)
            out.write(banner.encode("ascii"))
            out.write(b"\n")
            out.write(headers.as_bytes())

        def write(self, buf):
            return self._out.write(buf)

        def writable(self):
            return True

    class BinReader(io.RawIOBase):
        """Inverse of BinWriter: read and validate magic and headers, then act
        as any binary input stream.  Assume we have already read and validated
        the 4-byte magic number.  The banner and headers are available as
        attributes."""

        def __init__(self, inp, magic: bytes):
            super().__init__()
            assert magic == FileFormats.MAGIC
            assert inp.readable()
            self._inp = inp
            self.banner = inp.readline().strip().decode("ascii")
            self.headers = parse_http_headers(inp)

        def read(self, count):
            return self._inp.read(count)

        def readinto(self, buf):
            return self._inp.readinto(buf)

        def readable(self):
            return True

    class BannerWriter(io.RawIOBase):
        """Inspired by the GPG/PEM armored formats, this binary output stream
        adds BEGIN/END markers, headers, and produces line-wrapped ASCII using
        an encoding (base64 or quoted-printable) specified in subclasses."""

        def __init__(
            self, out: io.TextIOBase, banner: str, headers: HTTPMessage
        ):
            super().__init__()
            assert out.writable()
            self._out = out
            self._banner = banner
            self._write_banner("BEGIN")
            if len(headers) > 0:
                self._out.write(headers.as_string())

        def _write_banner(self, prefix: str):
            line = f"{FileFormats.DASHES}{prefix} {self._banner}{FileFormats.DASHES}"
            endl = "-\n" if len(line) % 4 == 0 else "\n"
            self._out.write(line)
            self._out.write(endl)

        def close(self):
            self._write_banner("END")
            self._out.flush()
            super().close()

        def writable(self):
            return True

    class BannerReader(ABC, io.RawIOBase):
        """Inspired by the GPG/PEM armored formats, this binary input stream
        notices BEGIN/END markers, headers, and line-wrapped ASCII using an
        encoding (base64 or quoted-printable) specified in subclasses."""

        header_re = re.compile(rb"^[-\w]+: ")

        def __init__(self, inp: io.TextIOWrapper, magic: bytes = b"----"):
            super().__init__()
            assert len(magic) == len(FileFormats.MAGIC)
            # Look for BEGIN banner, but continue in binary mode for now.
            dashes = FileFormats.DASHES.encode("ascii")
            line = magic.strip() + inp.buffer.readline().strip()
            while not line.startswith(dashes):
                line = inp.buffer.readline()
                if line is None or len(line) == 0:
                    raise ValueError("Could not find BEGIN banner")
                line = line.strip()
            line = line.strip(b"-")
            if not line.startswith(b"BEGIN "):
                raise ValueError("Malformed BEGIN banner")
            self.banner = line[6:].decode("ascii")
            # Accumulate data for parsing headers. This would be easier
            # except that I don't want to require a blank line in the
            # case of no headers.
            line = inp.buffer.readline()
            if self.header_re.match(line):  # There are headers
                headbuf = io.BytesIO()
                headbuf.write(line)
                while line != b"\n":
                    line = inp.buffer.readline()
                    headbuf.write(line)
                headbuf.seek(0)
                self.headers = parse_http_headers(headbuf)
                line = inp.buffer.readline()
            else:  # No headers
                self.headers = HTTPMessage()
            # Now we should already have our first line of base64 data.
            # Can continue reading in text mode, for better buffering.
            if line.strip(b"-").startswith(b"END"):
                self._buf = b""
            else:
                self._buf = self._decode_line(line)
            self._idx = 0
            self._inp = inp

        @abstractmethod
        def _decode_line(self, line: bytes):  # pragma: no cover
            ...

        def readable(self):
            return True

        def readinto(self, buf):
            idx = 0
            if len(self._buf) == 0:
                return 0
            while idx < len(buf):
                count = min(len(buf) - idx, len(self._buf) - self._idx)
                buf[idx : idx + count] = self._buf[
                    self._idx : self._idx + count
                ]
                idx += count
                self._idx += count
                if self._idx == len(self._buf):
                    line = self._inp.readline()
                    self._idx = 0
                    if line.strip("-").startswith("END"):
                        self._buf = b""
                        return idx
                    self._buf = self._decode_line(line)
            return len(buf)

    class ArmorWriter(BannerWriter):
        "Write binary data to line-wrapped base64."

        CHARS_PER_LINE = 48

        def __init__(self, out, banner: str, headers):
            super().__init__(out, banner, headers)
            self._buf = io.BytesIO()
            self._bytes_per_line = self.CHARS_PER_LINE * 3 // 4

        def write(self, buf):
            idx = 0
            while idx < len(buf):
                idx += self._write_from(buf, idx)
            return idx

        def _write_from(self, buf, idx):
            avail = len(buf) - idx
            capacity = self._bytes_per_line - self._buf.tell()
            if avail < capacity:
                self._buf.write(buf[idx:])
                return avail
            self._buf.write(buf[idx : idx + capacity])
            self._dump_buffer()
            return capacity

        def close(self):
            self._buf.truncate()
            if self._buf.tell() > 0:
                self._dump_buffer()
            super().close()

        def _dump_buffer(self):
            self._out.write(
                base64.b64encode(self._buf.getvalue()).decode("ascii")
            )
            self._out.write("\n")
            self._buf.seek(0)

    class ArmorReader(BannerReader):
        "Read armored-format Base64 data."

        def _decode_line(self, line):
            return base64.b64decode(line.strip())

    class QuotedPrintableWriter(BannerWriter):
        "Armor using quoted-printable encoding, so ASCII is human-readable."

        def __init__(self, out: io.TextIOBase, headers: HTTPMessage):
            super().__init__(out, FileFormats.QUO_PRI, headers)
            self._protect_first_line = len(headers) == 0

        def write(self, buf):
            if len(buf) == 0:
                return 0
            data = quopri.encodestring(buf).decode("ascii")
            if self._protect_first_line:
                # If there are no headers but the first line looks like one,
                # then the reader will get confused. Help it out by quoting the
                # first colon. The '=' won't match the header name regex and QP
                # decoding will restore the colon.
                eol = data.find("\n")
                colon = data.find(":")
                if colon >= 0 and (eol == -1 or colon < eol):
                    data = data.replace(":", "=3A", 1)
                    self._protect_first_line = False
                elif eol >= 0:
                    self._protect_first_line = False
            self._out.write(data)
            if not data.endswith("\n"):
                # This is lazy, but rather than letting lines get long, we'll
                # wrap after each write. The encodestring adds its own newlines,
                # but for many small writes this will look bad, but will work.
                self._out.write("=\n")
            return len(buf)

    class QuotedPrintableReader(BannerReader):
        "Read armored-format quoted-printable data."

        def _decode_line(self, line):
            return quopri.decodestring(line)

    @classmethod
    def writer(cls, banner: str, args: Namespace) -> ArmorWriter | BinWriter:
        "Configure a writer onto the given OUT stream."
        if getattr(args, "output", None) is None:
            args.output = sys.stdout
        if not hasattr(args, "headers"):
            args.headers = HTTPMessage()
        if getattr(args, "note", None) is not None:
            args.headers["Note"] = args.note
        if "Generated" not in args.headers:
            args.headers["Generated"] = str(datetime.utcnow())
        if getattr(args, "binary", False):
            return cls.BinWriter(args.output.buffer, banner, args.headers)
        return cls.ArmorWriter(args.output, banner, args.headers)

    @classmethod
    def reader(cls, src: t_source) -> ArmorReader | BinReader:
        "Configure a reader that uses the given input stream."
        instream = cls.wrapped_io(src)
        magic = instream.buffer.read(len(cls.MAGIC))
        if magic == cls.MAGIC:
            return cls.BinReader(instream.buffer, magic)
        return cls.ArmorReader(instream, magic)

    @classmethod
    def write_key(cls, key: int, args):
        "Write a key value to a stream."
        if getattr(args, "headers", None) is None:
            args.headers = HTTPMessage()
        args.headers["Bits"] = str(getattr(args, "bits", key.bit_length()))
        with cls.writer(cls.KEY, args) as out:
            cls.write_int(key, out)

    @classmethod
    def read_key(cls, src):
        "Read a key value from a stream."
        with cls.reader(src) as inp:
            assert inp.banner == cls.KEY, inp.banner
            return inp.headers, cls.read_int(inp)


####################################################################
###                                                          RSA ###


def is_prime(num, trials=20, randbelow=secrets.randbelow):
    """This is the Miller-Rabin (probabilistic) primality test.

    >>> [p for p in range(500,600) if is_prime(p)]
    [503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599]
    """
    if num < 2:
        return False
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if num % prime == 0:
            return num == prime
    sss, ddd = 0, num - 1
    while ddd % 2 == 0:
        sss, ddd = sss + 1, ddd >> 1
    for _ in range(trials):
        trial = 2 + randbelow(num - 2)
        yyy = pow(trial, ddd, num)
        if yyy in (1, num - 1):
            continue
        for _ in range(1, sss):
            yyy = (yyy * yyy) % num
            if yyy == 1:
                return False
            if yyy == num - 1:
                break
        else:
            return False
    return True


def next_prime(num: int) -> int:
    """Find the next prime larger than NUM.

    >>> next_prime(114)
    127
    """
    num += 1 + (num % 2)
    while not is_prime(num):
        num += 2
    return num


def inverse(val, mod):
    "Multiplicative modular inverse using extended Euclidean algorithm."
    aaa = mod
    bbb = val
    rem = -1
    tt1, tt2 = 0, 1
    while rem != 0:
        quo = aaa // bbb  # Integer division
        rem = aaa % bbb
        tt3 = tt1 - quo * tt2
        # print(f"{aaa:5} {bbb:5} {quo:5} {rem:5} {tt1:5} {tt2:5} {tt3:5}")
        aaa, bbb, tt1, tt2 = bbb, rem, tt2, tt3
    assert aaa == 1, f"NO INVERSE, GCD is {aaa}"
    return tt1 % mod


class RsaPublicKey:
    "An RSA public key, which consists of modulus and public exponent."

    BANNER = "RSAISH PUBLIC KEY"

    def __init__(
        self,
        mod: int,
        /,
        exp: int = 65537,
        *,
        gen: t.Optional[int] = None,
        who: t.Optional[str] = None,
    ):
        self.mod = mod
        self.exp = exp
        self.gen = gen  # generated timestamp
        self.who = who  # owner name

    def __eq__(self, other):
        return (
            isinstance(other, RsaPublicKey)
            and self.mod == other.mod
            and self.exp == other.exp
            and self.gen == other.gen
            and self.who == other.who
        )

    @property
    def nbytes(self):
        "The number of bytes required to represent the modulus."
        return math.ceil(self.mod.bit_length() / 8)

    def __str__(self):
        with io.StringIO() as buf:
            self.write_to(buf)
            return buf.getvalue()

    def __repr__(self):
        with io.StringIO() as buf:
            buf.write("RsaPublicKey")
            if self.mod.bit_length() < 100:
                buf.write(f"(0x{self.mod:x}")
                self.repr_attrs(buf)
            else:
                buf.write('.read_from("""\n')
                self.write_to(buf)
                buf.write('""")\n')
            return buf.getvalue()

    def repr_attrs(self, buf: io.StringIO):
        "Write remaining attributes (EXP, GEN, WHO) to BUF."
        buf.write(f", {self.exp}")
        if self.gen is not None:
            buf.write(f", gen={self.gen}")
        if self.who is not None:
            buf.write(f", who={repr(self.who)}")
        buf.write(")")

    def write_to(self, stream: io.TextIOBase):
        "Write the public key information in armored text format."
        args = Namespace(output=stream, headers=HTTPMessage())
        self.meta_to_headers(args.headers)
        with FileFormats.writer(self.BANNER, args) as out:
            FileFormats.write_int(self.exp, out)
            FileFormats.write_int(self.mod, out)

    def meta_to_headers(self, headers: HTTPMessage):
        "Save some metadata attributes (BITS, WHO, GEN) to HTTP-style HEADERS."
        headers["Bits"] = str(self.mod.bit_length())
        if self.who is not None:
            headers["Owner"] = self.who
        if self.gen is None:
            self.gen = int(time.time())
        headers["Generated"] = str(
            datetime.fromtimestamp(self.gen, timezone.utc)
        )

    def headers_to_meta(self, headers: HTTPMessage):
        "Retrieve metadata attributes (WHO, GEN) from HEADERS."
        self.who = headers.get("Owner")
        gen = headers.get("Generated")
        if gen is not None:
            self.gen = int(datetime.fromisoformat(gen).timestamp())

    @classmethod
    def read_from(cls, src: FileFormats.t_source):
        "Read a public key (armored format) from SRC."
        with FileFormats.reader(src) as inp:
            assert inp.banner == cls.BANNER
            exp = FileFormats.read_int(inp)
            mod = FileFormats.read_int(inp)
            result = RsaPublicKey(mod, exp=exp)
            result.headers_to_meta(inp.headers)
            return result

    def verify(
        self, buf: bytes, signature: int, hash_name: str = "sha256"
    ) -> bool:
        """Verify a signature.

        >>> kp = RsaKeyPair(65239, 55987)
        >>> kp.pub.verify(b"Hello!", 2538122000)
        True
        >>> kp.pub.verify(b"Hello!", 2538122100) # Modify signature
        False
        >>> kp.pub.verify(b"Hello.", 2538122000) # Modify buf
        False
        """
        mhash = hashlib.new(hash_name, buf).digest()
        ihash = int_from_bytes(mhash) % self.mod
        vhash = fastpow(signature, self.exp, modulo(self.mod))
        return ihash == vhash

    def encrypt_key(self, key):
        """Encrypt a number using this public key.  The number may then be used
        as a key in a symmetric cipher.  NOTE: This is so-called 'textbook' RSA
        encryption.  Even with suitably-sized keys, it is not sufficiently
        protected against malleability etc.

        >>> kp = RsaKeyPair(65239, 55987)
        >>> hex(kp.pub.encrypt_key(0xb0bacafe))
        '0xa8ac6b5e'

        """
        return fastpow(key, self.exp, modulo(self.mod))

    def encipher_bytes_to(self, buf: bytes, dest):
        """Encipher using a Salsa20 key protected by this RSA public key.
        It can then only be deciphered by the corresponding secret key.

        >>> kp = RsaKeyPair(65239, 55987)
        >>> buf = io.BytesIO()
        >>> kp.pub.encipher_bytes_to(b"Splendid", buf)
        >>> _ = buf.seek(0)
        >>> kp.decipher_bytes(buf)
        b'Splendid'
        """
        key = Salsa20.KEY.rand() % self.mod
        FileFormats.write_int(self.encrypt_key(key), dest)
        with Salsa20.Writer(dest, key) as out:
            out.write(buf)


class RsaKeyPair:
    "An object representing a public/secret key pair."

    BANNER = "RSAISH PRIVATE KEY PAIR"

    def __init__(
        self,
        prime1: int,
        prime2: int,
        /,
        exp: int = 65537,
        *,
        gen=None,
        who=None,
    ):
        assert is_prime(prime1), f"{prime1} is not prime"
        assert is_prime(prime2), f"{prime2} is not prime"
        self.pub = RsaPublicKey(prime1 * prime2, exp=exp, gen=gen, who=who)
        totient = (prime1 - 1) * (prime2 - 1)
        self.secret_exp = inverse(exp, totient)
        self.prime1 = prime1
        self.prime2 = prime2

    def __eq__(self, other):
        return (
            isinstance(other, RsaKeyPair)
            and self.pub == other.pub
            and self.prime1 == other.prime1
            and self.prime2 == other.prime2
            and self.secret_exp == other.secret_exp
        )

    def __str__(self):
        with io.StringIO() as buf:
            self.write_to(buf)
            return buf.getvalue()

    def __repr__(self):
        with io.StringIO() as buf:
            buf.write("RsaKeyPair")
            if self.pub.mod.bit_length() < 100:
                buf.write(f"({self.prime1}, {self.prime2}")
                self.pub.repr_attrs(buf)
            else:
                buf.write('.read_from("""\n')
                self.write_to(buf)
                buf.write('""")\n')
            return buf.getvalue()

    def write_to(self, stream: io.TextIOWrapper):
        "Write a key pair in armored format to STREAM."
        args = Namespace(output=stream, headers=HTTPMessage())
        self.pub.meta_to_headers(args.headers)
        with FileFormats.writer(self.BANNER, args) as out:
            FileFormats.write_int(self.pub.exp, out)
            FileFormats.write_int(self.prime1, out)
            FileFormats.write_int(self.prime2, out)

    @classmethod
    def read_from(cls, src: FileFormats.t_source):
        "Read a key pair in armored format from SRC."
        with FileFormats.reader(src) as inp:
            assert inp.banner == cls.BANNER
            exp = FileFormats.read_int(inp)
            prime1 = FileFormats.read_int(inp)
            prime2 = FileFormats.read_int(inp)
            result = RsaKeyPair(prime1, prime2, exp=exp)
            result.pub.headers_to_meta(inp.headers)
            return result

    @classmethod
    def generate(cls, nbits: int = 2048, exp: int = 65537, who=None):
        "Generate a key pair using random primes of about NBITS."
        limit = 1 << (nbits // 2)
        prime1 = next_prime(secrets.randbelow(limit))
        prime2 = next_prime(secrets.randbelow(limit))
        return RsaKeyPair(
            prime1, prime2, exp=exp, who=who, gen=int(time.time())
        )

    def sign(self, buf: bytes, hash_name: str = "sha256") -> int:
        """Sign the hash of a byte string.

        >>> kp = RsaKeyPair(65239, 55987)
        >>> kp.sign(b"Hello!")
        2538122000
        """
        mhash = hashlib.new(hash_name, buf).digest()
        ihash = int_from_bytes(mhash) % self.pub.mod
        return fastpow(ihash, self.secret_exp, modulo(self.pub.mod))

    def decrypt_key(self, key):
        """Decrypt a number using this secret key.

        >>> kp = RsaKeyPair(65239, 55987)
        >>> hex(kp.decrypt_key(0xa8ac6b5e))
        '0xb0bacafe'
        """
        return fastpow(key, self.secret_exp, modulo(self.pub.mod))

    def decipher_bytes(self, src):
        """Decipher using an embedded Salsa20 key protected by this RSA pair.

        >>> kp = RsaKeyPair(54401, 56681)
        >>> buf = io.BytesIO()
        >>> kp.pub.encipher_bytes_to(b"Elementary", buf)
        >>> _ = buf.seek(0)
        >>> kp.decipher_bytes(buf)
        b'Elementary'
        """
        key = self.decrypt_key(FileFormats.read_int(src))
        with Salsa20.Reader(src, key) as inp:
            return inp.read()


class RSA:
    """Top-level API for 'textbook' RSA.

    >>> alice = RsaKeyPair(50417, 55259, who="Alice")
    >>> bob = RsaKeyPair(50753, 49871, who="Bob")
    >>> eve = RsaKeyPair(54401, 56681, who="Eve")

    >>> buf = FileFormats.wrapped_io()
    >>> RSA.sign_encrypt_text(bob, alice.pub, "Hey, Alice!", buf)

    >>> _ = buf.seek(0)
    >>> RSA.decrypt_print_text(bob.pub, alice, buf)
    <BLANKLINE>
    NOTE: Valid signature from Bob
    <BLANKLINE>
    Hey, Alice!
    <BLANKLINE>

    >>> _ = buf.seek(0)
    >>> RSA.decrypt_print_text(bob.pub, eve, buf)
    <BLANKLINE>
    ERROR: Not a valid signature from Bob

    """

    BANNER_SIGN_CIPHER = "RSAISH SIGNED CIPHERTEXT"
    TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"

    @classmethod
    def sign_then_encrypt(
        cls,
        sender: RsaKeyPair,
        recipient: RsaPublicKey,
        data: bytes,
        dest: io.TextIOWrapper,
        typ: str = "application/octet-stream",
    ):
        "Sign with sender's secret key, then encrypt with recipient's public."
        args = Namespace(output=dest, headers=HTTPMessage())
        args.headers["From"] = sender.pub.who
        args.headers["To"] = recipient.who
        args.headers["Content-Type"] = typ
        with FileFormats.writer(cls.BANNER_SIGN_CIPHER, args) as out:
            FileFormats.write_int(sender.sign(data), out)
            recipient.encipher_bytes_to(data, out)

    @classmethod
    def decrypt_then_verify(
        cls, sender: RsaPublicKey, recipient: RsaKeyPair, src
    ):
        "Decrypt and verify a message created with sign_then_encrypt."
        with FileFormats.reader(src) as inp:
            assert inp.banner == cls.BANNER_SIGN_CIPHER
            sig = FileFormats.read_int(inp)
            data = recipient.decipher_bytes(inp)
            okay = sender.verify(data, sig)
            return okay, data, inp.headers

    @classmethod
    def sign_encrypt_text(
        cls,
        sender: RsaKeyPair,
        recipient: RsaPublicKey,
        message: str,
        dest=sys.stdout,
    ):
        "Sign and encrypt a UTF-8 encoded text message."
        cls.sign_then_encrypt(
            sender,
            recipient,
            message.encode("utf-8"),
            dest,
            typ=cls.TEXT_CONTENT_TYPE,
        )

    @classmethod
    def decrypt_print_text(
        cls, sender: RsaPublicKey, recipient: RsaKeyPair, src
    ):
        "Decrypt and verify a UTF-8 message created with sign_encrypt_text."
        okay, msg, headers = cls.decrypt_then_verify(sender, recipient, src)
        print()
        if not okay:
            print(f"ERROR: Not a valid signature from {sender.who}")
            return
        print(f"NOTE: Valid signature from {sender.who}")
        assert headers.get("Content-Type") == cls.TEXT_CONTENT_TYPE
        print()
        print(msg.decode("utf-8"))
        print()

    @classmethod
    def sign_only_text(cls, signer: RsaKeyPair, message: str, dest=sys.stdout):
        """Create a signed (but not encrypted) text message. In this format, the
        message is shown with quoted-printable encoding and the signature is
        contained in headers.

        >>> alice = RsaKeyPair(50417, 55259, who="Alice")
        >>> RSA.sign_only_text(alice, "Howdy!", sys.stdout)
        -----BEGIN QUOTED PRINTABLE TEXT-----
        Content-Type: text/plain; charset=utf-8
        Content-Transfer-Encoding: quoted-printable
        Signature: w6QjVg==
        Signed-By: Alice
        <BLANKLINE>
        Howdy!=
        -----END QUOTED PRINTABLE TEXT-----

        """
        data = message.encode("utf-8")
        sig = base64.b64encode(
            int_to_bytes(signer.sign(data), signer.pub.nbytes)
        ).decode("ascii")
        chunk = math.ceil(len(sig) / math.ceil(len(sig) / 60))
        headers = HTTPMessage()
        headers["Content-Type"] = cls.TEXT_CONTENT_TYPE
        headers["Content-Transfer-Encoding"] = "quoted-printable"
        for idx in range(0, len(sig), chunk):
            headers["Signature"] = sig[idx : idx + chunk]
        headers["Signed-By"] = signer.pub.who
        with FileFormats.QuotedPrintableWriter(dest, headers) as out:
            out.write(data)

    @classmethod
    def verify_signed_text(cls, signer: RsaPublicKey, src):
        """Verify a signed text message in the quoted-printable format.

        >>> alice = RsaKeyPair(50417, 55259, who="Alice")
        >>> RSA.verify_signed_text(alice.pub, chr(10).join([
        ...   "-----BEGIN QUOTED PRINTABLE TEXT-----",
        ...   "Content-Type: text/plain; charset=utf-8",
        ...   "Signature: w6QjVg==",
        ...   "",
        ...   "Howdy!=",
        ...   "-----END QUOTED PRINTABLE TEXT-----"]))
        <BLANKLINE>
        NOTE: Valid signature from Alice
        <BLANKLINE>
        Howdy!
        <BLANKLINE>

        >>> alice = RsaKeyPair(50417, 55259, who="Alice")
        >>> RSA.verify_signed_text(alice.pub, chr(10).join([
        ...   "-----BEGIN QUOTED PRINTABLE TEXT-----",
        ...   "Content-Type: text/plain; charset=utf-8",
        ...   "Signature: w6QjVg==",
        ...   "",
        ...   "Howdy..=",
        ...   "-----END QUOTED PRINTABLE TEXT-----"]))
        <BLANKLINE>
        ERROR: Not a valid signature from Alice
        """
        instream = FileFormats.wrapped_io(src)
        with FileFormats.QuotedPrintableReader(instream) as inp:
            data = inp.read()
            assert data is not None
            sig = int_from_bytes(
                base64.b64decode("".join(inp.headers.get_all("Signature", [])))
            )
        okay = signer.verify(data, sig)
        print()
        if not okay:
            print(f"ERROR: Not a valid signature from {signer.who}")
            return
        print(f"NOTE: Valid signature from {signer.who}")
        print()
        print(data.decode("utf-8"))
        print()


####################################################################
###                                       COMMAND LINE INTERFACE ###


def run_rand(args):
    "Generate a random number for use as a key."
    if isinstance(args.bits, str):
        args.bits = int(args.bits, base=0)
    key = secrets.randbelow(2**args.bits)
    if (
        getattr(args, "binary", False)
        or getattr(args, "output", None) is not None
    ):
        FileFormats.write_key(key, args)
        if not getattr(args.output, "name", "<mem>").startswith("<"):
            print(f"Saved to {args.output.name}", file=sys.stderr)
    else:
        print(hex(key))


def add_rand_args(cmdp):
    "Configure argument parser for rand command."
    argp = cmdp.add_parser(
        "rand",
        help=run_rand.__doc__,
        description=run_rand.__doc__,
    )
    argp.set_defaults(func=run_rand)
    argp.add_argument(
        "bits",
        metavar="BITS",
        default=Salsa20.KEY.BITS,
        help=f"size of random number, default is {Salsa20.KEY.BITS}",
        nargs="?",
    )
    add_output_args(argp)


def add_output_args(argp):
    "Configure output-related args for an argument parser."
    argp.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        metavar="FILE",
        help="save to FILE ('-' for stdout)",
    )
    argp.set_defaults(headers=HTTPMessage())
    argp.add_argument(
        "--binary", "-b", action="store_true", help="output in binary mode"
    )
    argp.add_argument(
        "--note",
        "-n",
        metavar="TEXT",
        help="add TEXT to armored output headers (implies -o)",
    )


def run_s20enc(args):
    "Encrypt using Salsa20 stream cipher."
    if getattr(args, "keyfile", None) is not None:
        key_headers, args.key = FileFormats.read_key(args.keyfile)
        if getattr(args, "headers", None) is None:
            args.headers = HTTPMessage()
        for key_header in key_headers:
            args.headers[f"Key-{key_header}"] = key_headers[key_header]
    elif isinstance(args.key, str):
        args.key = int(args.key, base=0)
    if getattr(args, "message", None) is not None:
        args.input = io.BytesIO(args.message.encode())
    elif getattr(args, "input", None) is None:
        args.input = sys.stdin.buffer
    elif hasattr(args.input, "name"):
        args.headers["Original-File"] = args.input.name
    with FileFormats.writer(FileFormats.S20_DATA, args) as out:
        Salsa20.encipher_copy_stream(args.input, out, args.key)


def add_s20enc_args(cmdp):
    "Configure argument parser for s20enc command."
    argp = cmdp.add_parser(
        "s20enc",
        help=run_s20enc.__doc__,
        description=run_s20enc.__doc__,
    )
    argp.set_defaults(func=run_s20enc)
    argp_inp = argp.add_mutually_exclusive_group()
    argp_inp.add_argument(
        "--message",
        "-m",
        metavar="TEXT",
        help="message to encrypt, or specify input file",
    )
    argp_inp.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        type=argparse.FileType("rb"),
        help="file to encrypt, or use standard input",
    )
    add_key_args(argp)
    add_output_args(argp)


def add_key_args(argp):
    "Configure key-related args for an argument parser."
    argp_key = argp.add_mutually_exclusive_group(required=True)
    argp_key.add_argument(
        "key",
        metavar="KEY",
        nargs="?",
        help="numerical representation of key (include '0x' prefix)",
    )
    argp_key.add_argument(
        "--keyfile",
        "-k",
        type=argparse.FileType("r"),
        metavar="FILE",
        help="read armored key from FILE (instead of numerical KEY)",
    )


def run_s20dec(args):
    "Decrypt using Salsa20 stream cipher."
    if getattr(args, "keyfile", None) is not None:
        _, args.key = FileFormats.read_key(args.keyfile)
    elif isinstance(args.key, str):
        args.key = int(args.key, base=0)
    if getattr(args, "input", None) is None:
        args.input = sys.stdin
    if getattr(args, "output", None) is None:
        args.output = sys.stdout.buffer
    with FileFormats.reader(args.input) as inp:
        assert inp.banner == FileFormats.S20_DATA
        Salsa20.decipher_copy_stream(inp, args.output, args.key)


def add_s20dec_args(cmdp):
    "Configure argument parser for s20dec command."
    argp = cmdp.add_parser(
        "s20dec",
        help=run_s20dec.__doc__,
        description=run_s20dec.__doc__,
    )
    argp.set_defaults(func=run_s20dec)
    argp.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="file to decrypt, or use standard input",
    )
    argp.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wb"),
        metavar="FILE",
        help="save to FILE ('-' for stdout)",
    )
    add_key_args(argp)


def parse_args(*args, **kwargs):
    "Parse arguments for command-line interface."
    argp = argparse.ArgumentParser(
        prog="ai265",
        description="Utilities and demos for Modern Cryptography, LIU Spring 24.",
        epilog="Please do not rely on these experiments for actual privacy!",
    )
    argp.set_defaults(func=None, prog=argp.prog, failfast=False)
    argp.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable informative (-v) or debugging (-vv) messages",
    )
    cmdp = argp.add_subparsers()
    add_rand_args(cmdp)
    add_s20enc_args(cmdp)
    add_s20dec_args(cmdp)

    argp_test_help = "Run unit tests and doc tests."
    argp_test = cmdp.add_parser(
        "test", help=argp_test_help, description=argp_test_help
    )
    argp_test.add_argument(
        "--failfast",
        "-f",
        action="store_true",
        help="stop the test run on the first failure",
    )
    return argp.parse_args(*args, **kwargs)


def setup_logging(verbose: int = 0, **_kwargs):
    "Configure log level based on verbose argument."
    logging.basicConfig()
    logging.root.name = ""
    if verbose == 2:
        logging.root.setLevel(logging.DEBUG)
        logging.debug("Logging enabled")
    elif verbose == 1:
        logging.root.setLevel(logging.INFO)
        logging.info("Logging enabled")


@contextmanager
def capture_output(name="stdout"):
    "Capture standard output to a string buffer."
    old = getattr(sys, name)
    buf = io.StringIO()
    try:
        setattr(sys, name, buf)
        yield buf
    finally:
        setattr(sys, name, old)


@contextmanager
def capture_stdout_bytes():
    "Capture standard output to a byte buffer."
    old_stdout = sys.stdout
    buf = io.BytesIO()
    buf.close = lambda: None
    try:
        sys.stdout = FileFormats.wrapped_io(buf)
        yield buf
    finally:
        sys.stdout = old_stdout


@contextmanager
def provide_stdin_bytes(buf: bytes):
    "Provide a byte buffer to act as standard input."
    old_stdin = sys.stdin
    try:
        sys.stdin = FileFormats.wrapped_io(buf)
        yield
    finally:
        sys.stdin = old_stdin


@contextmanager
def capture_logs():
    "Temporarily capture logs to a string buffer and restore log level after."
    old_log_stream = logging.root.handlers[0].stream
    old_log_level = logging.root.getEffectiveLevel()
    buf = io.StringIO()
    try:
        logging.root.handlers[0].setStream(buf)
        yield buf
    finally:
        logging.root.handlers[0].setStream(old_log_stream)
        logging.root.setLevel(old_log_level)


####################################################################
###                                    MAIN BLOCK AND UNIT TESTS ###


def load_tests(_loader, tests, _ignore):
    "Integrate doc tests into unittest discovery."
    tests.addTests(doctest.DocTestSuite())
    return tests


if __name__ == "__main__":
    cli_args = parse_args()
    setup_logging(**vars(cli_args))
    logging.debug(cli_args)
    if cli_args.func is not None:
        sys.exit(cli_args.func(cli_args))  # pragma: no cover

    from hypothesis import assume, given, strategies as st
    import Crypto.Cipher.AES
    import hmac
    import string

    class TestAlphabeticCiphers(unittest.TestCase):
        "Test alphabetic cipher tools."

        @given(st.characters(), st.integers())
        def test_rotate_negative_inverts(self, char: str, offset: int):
            "Rotation by OFFSET and negative OFFSET should be inverses"
            self.assertEqual(
                rotate_letter(rotate_letter(char, offset), -offset), char
            )

        @given(st.characters(), st.integers())
        def test_rotate_plus_26(self, char: str, offset: int):
            "Rotation by OFFSET and OFFSET+26 should be the same"
            self.assertEqual(
                rotate_letter(char, offset), rotate_letter(char, offset + 26)
            )

        @given(st.characters(), st.integers())
        def test_rotate_same_case(self, char: str, offset: int):
            "Rotation should preserve case"
            self.assertEqual(
                char.isupper(), rotate_letter(char, offset).isupper()
            )
            self.assertEqual(
                char.islower(), rotate_letter(char, offset).islower()
            )

    class TestHMAC(unittest.TestCase):
        "Test our implementation of a hash-based message authentication code."

        @given(
            st.binary(min_size=12, max_size=64),
            st.binary(max_size=50),
            st.sampled_from(
                list(hashlib.algorithms_guaranteed - {"shake_128", "shake_256"})
            ),
        )
        # Avoid hashes that require .digest(length).
        def test_hash_based_auth_code(self, key, message, hash_name):
            "Test that our HMAC is equivalent to the Python's."
            mine = hash_based_auth_code(key, message, hash_name)
            theirs = hmac.new(key, message, hash_name).digest()
            self.assertEqual(mine, theirs)

        @given(
            st.binary(min_size=60),
            st.binary(),
        )
        def test_hash_based_auth_code_long_key(self, key, message):
            "Test that our HMAC is equivalent to the Python's."
            mine = hash_based_auth_code(key, message, "sha1")
            theirs = hmac.new(key, message, "sha1").digest()
            self.assertEqual(mine, theirs)

    class TestMath(unittest.TestCase):
        "Test our operators for multiplication, exponentiation, and modulus."

        @given(st.integers(), st.integers(min_value=0, max_value=1 << 9))
        def test_fastpow(self, base, exp):
            "Ensure that fastpow is equivalent to '**'."
            self.assertEqual(fastpow(base, exp), base**exp)

        @given(
            st.integers(),
            st.integers(min_value=0, max_value=1 << 9),
            st.integers(),
        )
        def test_fastpow_mod(self, base, exp, mod):
            "Test fastpow using modulo is equivalent to '**' and '%'."
            assume(mod != 0)
            self.assertEqual(fastpow(base, exp, modulo(mod)), base**exp % mod)

        @given(st.integers(min_value=0), st.integers())
        def test_bin_mul(self, fst, snd):
            "Ensure that bin_mul is equivalent to '*'"
            self.assertEqual(bin_mul(fst, snd), fst * snd)

        @given(st.integers(min_value=0), st.integers(), st.integers())
        def test_bin_mul_mod(self, fst, snd, mod):
            "Ensure that bin_mul with modular add/double matches '*' with '%'."
            assume(mod != 0)
            self.assertEqual(
                bin_mul(fst, snd, mod=modulo(mod)),
                fst * snd % mod,
            )

        @given(st.integers(min_value=0))
        def test_bin_mul_xor_5c(self, val):
            "Try using carryless (xor) multiplication against a known value."
            factor = 0x5C
            truth = val << 2 ^ val << 3 ^ val << 4 ^ val << 6
            self.assertEqual(bin_mul(factor, val, add=operator.xor), truth)
            self.assertEqual(bin_mul(val, factor, add=operator.xor), truth)

        @given(st.integers(min_value=0))
        def test_bin_mul_xor_c09(self, val):
            "Try using carryless (xor) multiplication against a known value."
            factor = 0xC09
            truth = val ^ val << 3 ^ val << 10 ^ val << 11
            self.assertEqual(bin_mul(factor, val, add=operator.xor), truth)
            self.assertEqual(bin_mul(val, factor, add=operator.xor), truth)

        @given(st.data())
        def test_xor_mod(self, data):
            "Test xor_mod against xor multiplication."
            mod = data.draw(st.integers(min_value=2))
            const = data.draw(st.integers(min_value=0))
            rem = data.draw(st.integers(min_value=0, max_value=mod // 2))
            self.assertEqual(
                xor_mod(mod)(bin_mul(mod, const, operator.xor)) ^ rem, rem
            )

    class TestWords(unittest.TestCase):
        "Test the fixed-size word classes."

        @staticmethod
        def words(word) -> st.SearchStrategy[int]:
            """Hypothesis strategy for a fixed-size integer class.  Generating
            byte strings first then converting to int seems to give more
            diversity of values, but is still shrinkable to zero."""
            return st.binary(min_size=word.BYTES, max_size=word.BYTES).map(
                lambda b: word.from_bytes(b) & word.MAX
            )

        octets = words(Word8)

        @given(words(Word64))
        def test_split_then_join(self, num: int):
            "Test Word64 <-> Word16s round-trip"
            self.assertEqual(num, Word16.join(Word64.split(num, Word16)))

        @given(st.lists(words(Word16), min_size=4, max_size=4))
        def test_join_then_split(self, nums: list[int]):
            "Test Word16s <-> Word64 round-trip"
            self.assertEqual(
                nums, list(Word64.split(Word16.join(nums), Word16))
            )

        @given(words(Word64))
        def test_to_from_bytes(self, num: int):
            "Test Word64 <-> bytes round-trip"
            self.assertEqual(num, Word64.from_bytes(Word64.to_bytes(num)))

        @given(st.binary(min_size=8, max_size=8))
        def test_from_to_bytes(self, buf: bytes):
            "Test bytes <-> Word64 round-trip"
            self.assertEqual(buf, Word64.to_bytes(Word64.from_bytes(buf)))

    class AbstractGroupoidTest(ABC):
        "Base class for testing group laws."
        assertEqual: t.Any
        groups: st.SearchStrategy

        @staticmethod
        @abstractmethod
        def values(group, count: int = 1) -> st.SearchStrategy:
            "Produce COUNT values from GROUP."

    def several(gen, count: int = 1):
        "Helper for generation COUNT values from GEN."
        return gen if count == 1 else st.tuples(*[gen] * count)

    class ClosedMagmaLaw(AbstractGroupoidTest):
        "A magma is a groupoid with a closed binary operator."

        @given(st.data())
        def test_closed(self, data):
            "Test that operator is closed wrt the group."
            grp = data.draw(self.groups)
            aaa, bbb = data.draw(self.values(grp, 2))
            assert aaa in grp, aaa
            assert bbb in grp, bbb
            ccc = grp(aaa, bbb)
            assert ccc in grp, ccc

    class CommutativeLaw(AbstractGroupoidTest):
        "A group is commutative (Abelian) if for any a@b = b@a."

        @given(st.data())
        def test_commutative(self, data):
            "Test that operator is commutative."
            grp = data.draw(self.groups)
            aaa, bbb = data.draw(self.values(grp, 2))
            self.assertEqual(grp(aaa, bbb), grp(bbb, aaa))

    class AssociativeLaw(AbstractGroupoidTest):
        "Group operators should be associative."

        @given(st.data())
        def test_associative(self, data):
            "Test that operator is associative."
            grp = data.draw(self.groups)
            aaa, bbb, ccc = data.draw(self.values(grp, 3))
            self.assertEqual(grp(aaa, grp(bbb, ccc)), grp(grp(aaa, bbb), ccc))

    class IdentityLaws(AbstractGroupoidTest):
        "Groupoids may distinguish an identity element subject to these laws."

        @given(st.data())
        def test_contains_identity(self, data):
            "Test that the identity element is a member of the group."
            grp = data.draw(self.groups)
            assert grp.identity in grp

        @given(st.data())
        def test_left_identity(self, data):
            "Test that applying identity on the left leaves value unchanged."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            self.assertEqual(grp(grp.identity, val), val)

        @given(st.data())
        def test_right_identity(self, data):
            "Test that applying identity on the right leaves value unchanged."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            self.assertEqual(grp(val, grp.identity), val)

    class InverseLaws(AbstractGroupoidTest):
        "A group requires that every element has an inverse."

        @given(st.data())
        def test_inverse_closed(self, data):
            "Test that a value's inverse is a member of the group."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            inv = grp.inverse(val)
            assert inv in grp, inv

        @given(st.data())
        def test_left_inverse(self, data):
            "Test that applying inverse on the left produces identity."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            self.assertEqual(grp(grp.inverse(val), val), grp.identity)

        @given(st.data())
        def test_right_inverse(self, data):
            "Test that applying inverse on the right produces identity."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            self.assertEqual(grp(val, grp.inverse(val)), grp.identity)

        @given(st.data())
        def test_twice_inverse(self, data):
            "Test that applying inverse twice produces the original value."
            grp = data.draw(self.groups)
            val = data.draw(self.values(grp))
            self.assertEqual(grp.inverse(grp.inverse(val)), val)

    class MonoidLaws(ClosedMagmaLaw, AssociativeLaw, IdentityLaws):
        "A monoid is a closed associative groupoid with an identity element."

    class GroupLaws(MonoidLaws, InverseLaws):
        "A group is a monoid with an inverse operator."

    class TestXorZip(unittest.TestCase, CommutativeLaw, GroupLaws):
        """Test that xor_zip implements an Abelian group over iterable octets,
        where equivalence is up to the shortest length."""

        t_octets = list[int] | bytes

        groups = st.just(xor_zip)

        @staticmethod
        def values(group: list[int] | bytes, count=1):
            return several(
                st.one_of(st.lists(TestWords.octets), st.binary()), count
            )

        def assertEqual(self, first, second, msg=None):
            """A flexible notion of equality for this group.  FIRST and SECOND
            could be different types of iterables, with different lengths.
            Convert to the same type, and check up to the minimum length."""
            vals = list(zip(first, second))
            fst = [a for a, _ in vals]
            snd = [b for _, b in vals]
            return super().assertEqual(fst, snd, msg)

        def test_xor_group_str_containment(self):
            "Character strings are not members of the xor_zip group."
            assert "abc" not in xor_zip

        def test_xor_group_non_w8_containment(self):
            "Iterables with non-byte values are not in the xor_zip group."
            assert [13, 26, 52] in xor_zip
            assert [13, 256, 392] not in xor_zip

    class TestPrimes(unittest.TestCase):
        "Test the primality test."

        @given(st.integers(min_value=2), st.integers(min_value=2))
        def test_composite(self, ppp, qqq):
            "Ensure that a composite number P*Q is not labeled as prime."
            assert not is_prime(ppp * qqq)

        @given(st.integers(min_value=3, max_value=1 << 20))
        def test_factorable_or_not(self, num):
            "Try to factor a smallish number, ensure result matches is_prime."
            for cand in range(2, math.isqrt(num) + 1):
                if num % cand == 0:
                    assert not is_prime(num)  # Found a factor
                    return
            assert is_prime(num)  # No factor less than sqrt

        def test_one_not_prime(self):
            "One isn't prime."
            assert not is_prime(1)

        @given(st.data())
        def test_special_composite(self, data):
            "These numbers hit a special case in is_prime."
            assert not is_prime(
                8321,
                randbelow=lambda x: data.draw(
                    st.sampled_from([7204, 5530, 4454, 3748])
                ),
            )

    class TestS20(unittest.TestCase):
        "Test Salsa20 cipher."

        keys = TestWords.words(Salsa20.KEY)
        nonces = TestWords.words(Salsa20.NONCE)
        offsets = TestWords.words(Word70)

        @given(keys, nonces, st.binary(max_size=io.DEFAULT_BUFFER_SIZE))
        def test_encipher_decipher_stream(self, key, nonce, msg):
            "Round-trip for encipher/decipher byte strings"
            msg *= 40
            self.assertEqual(
                Salsa20.decipher_bytes(
                    Salsa20.encipher_bytes(msg, key, nonce), key
                ),
                msg,
            )
            self.assertTrue(Salsa20.Reader.readable())
            self.assertTrue(Salsa20.Writer.writable())

        @given(
            keys,
            nonces,
            offsets,
        )
        def test_keystream_seek_then_tell(self, key, nonce, pos):
            "Seek to a position, then make sure tell() returns same position."
            with Salsa20(key, nonce) as kstream:
                assert kstream.seekable()
                assert kstream.readable()
                kstream.seek(pos)
                pos2 = kstream.tell()
                self.assertEqual(pos, pos2)

        @given(
            keys,
            nonces,
            offsets,
            st.integers(min_value=0, max_value=250),
        )
        def test_keystream_seek_absolute(self, key, nonce, pos, size):
            "Read the key stream, seek, and re-read"
            with Salsa20(key, nonce) as kstream:
                kstream.seek(pos)
                buf1 = bytearray(size)
                buf2 = bytearray(size)
                size1 = kstream.readinto(buf1)
                self.assertEqual(size1, size)
                kstream.seek(pos)
                size2 = kstream.readinto(buf2)
                self.assertEqual(size2, size)
                self.assertEqual(buf2, buf1)

        @given(
            keys,
            nonces,
            offsets,
            st.integers(min_value=0, max_value=250),
        )
        def test_keystream_seek_relative(self, key, nonce, pos1, size):
            "Read the key stream, seek negative to go back, and re-read."
            with Salsa20(key, nonce) as kstream:
                kstream.seek(pos1)
                buf1 = kstream.read(size)
                self.assertEqual(len(buf1), size)
                kstream.seek(-size, os.SEEK_CUR)
                pos2 = kstream.tell()
                self.assertEqual(pos2, pos1)
                buf2 = kstream.read(size)
                self.assertEqual(len(buf2), size)
                self.assertEqual(buf2, buf1)

    class TestAES(unittest.TestCase):
        """Test encoding and decoding with AES."""

        keys = st.sampled_from([16, 24, 32]).flatmap(
            lambda nbytes: st.binary(min_size=nbytes, max_size=nbytes)
        )

        def test_key_gen(self):
            "Try AES.key_gen, just to cover it."
            AES.key_gen()

        @given(keys, st.binary(min_size=16, max_size=16))
        def test_enc_block_vs_cryptodome(self, key, data):
            "Check that my encryption is the same as cryptodome's."
            mine = AES(key)
            theirs = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_ECB)
            self.assertEqual(mine.enc_block(data), theirs.encrypt(data))

        @given(keys, st.binary(min_size=16, max_size=16))
        def test_my_enc_their_dec(self, key, data):
            "Check that cryptodome can decrypt a block that I encrypted."
            cipher = AES(key).enc_block(data)
            theirs = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_ECB)
            plain = theirs.decrypt(cipher)
            self.assertEqual(data, plain)

    class TestFileFormats(unittest.TestCase):
        "Test reading and writing of binary/ASCII file formats."

        @staticmethod
        def make_headers(hlist):
            "Create headers from a list of pairs."
            headers = HTTPMessage()
            for name, val in hlist:
                headers[name] = val
            return headers

        st_header_name = st.text(string.ascii_letters, min_size=1)
        st_header_value = st.text(
            "".join(set(string.printable) - set(string.whitespace))
        )
        st_header = st.tuples(st_header_name, st_header_value)
        st_headers = st.lists(st_header).map(make_headers)

        @given(st.binary(min_size=1), st_headers)
        def test_armor_round_trip(self, data, headers):
            "Test a round-trip of binary data through ArmorWriter/ArmorReader."
            buf = FileFormats.wrapped_io()
            with FileFormats.ArmorWriter(buf, "TEST", headers) as out:
                out.write(data)
            buf.seek(0)
            with FileFormats.ArmorReader(buf) as inp:
                self.assertEqual(inp.banner, "TEST")
                self.assertEqual(inp.headers.as_string(), headers.as_string())
                result = inp.read()
            self.assertEqual(result, data)

        @given(st.lists(st.binary()), st_headers)
        def test_quoted_printable_round_trip(self, data, headers):
            "Round-trip of binary data in quoted-printable writer/reader."
            buf = FileFormats.wrapped_io()
            with FileFormats.QuotedPrintableWriter(buf, headers) as out:
                for bstr in data:
                    assume(13 not in bstr)
                    out.write(bstr)
            buf.seek(0)
            with FileFormats.QuotedPrintableReader(buf) as inp:
                self.assertEqual(inp.headers.as_string(), headers.as_string())
                result = inp.read()
            self.assertEqual(result, b"".join(data))

        @given(st.integers(min_value=2, max_value=1024), st.booleans())
        def test_key_writer_round_trip(self, nbytes, binary):
            "Test round-trip of a key to a file."
            buf = FileFormats.wrapped_io()
            key1 = secrets.randbelow(256**nbytes)
            FileFormats.write_key(key1, Namespace(output=buf, binary=binary))
            buf.seek(0)
            _, key2 = FileFormats.read_key(buf)
            self.assertEqual(key2, key1)

        def test_malformed_begin(self):
            "Trigger malformed-begin ValueError in ArmorReader."
            with self.assertRaises(ValueError):
                FileFormats.ArmorReader(
                    FileFormats.wrapped_io(b"-----BEGON-----\n")
                )

        def test_missing_banner(self):
            "Trigger missing-banner ValueError in ArmorReader."
            with self.assertRaises(ValueError):
                FileFormats.ArmorReader(FileFormats.wrapped_io(b"\n\nBEGON\n"))

    class TestRSA(unittest.TestCase):
        "Test the RSA operations."

        st_primes = st.integers(min_value=1000, max_value=1 << 60).map(
            next_prime
        )
        st_exp = st.sampled_from([65537, 3, 11, 17, 19])

        # Pick two primes and an exponent that are 'compatible'. That is,
        # EXP must be relatively prime with (P-1)*(Q-1).
        st_params = st.tuples(st_primes, st_primes, st_exp).filter(
            lambda x: math.gcd((x[0] - 1) * (x[1] - 1), x[2]) == 1
        )

        st_who = st.sampled_from([None, "Alice", "Bob", "Carla", "Dave"])
        st_gen = st.one_of(
            st.none(), st.datetimes().map(lambda d: int(d.timestamp()))
        )

        st_pubs = st_params.flatmap(
            lambda p: TestRSA.st_who.flatmap(
                lambda w: TestRSA.st_gen.map(
                    lambda g: RsaPublicKey(p[0] * p[1], p[2], who=w, gen=g)
                )
            )
        )

        st_keys = st_params.flatmap(
            lambda p: TestRSA.st_who.flatmap(
                lambda w: TestRSA.st_gen.map(
                    lambda g: RsaKeyPair(p[0], p[1], p[2], who=w, gen=g)
                )
            )
        )

        @given(st_pubs)
        def test_public_key_stream_round_trip(self, pkey1):
            "Test write_to/read_from for RsaPublicKey."
            buf = FileFormats.wrapped_io()
            pkey1.write_to(buf)
            buf.seek(0)
            pkey2 = RsaPublicKey.read_from(buf)
            self.assertEqual(pkey1, pkey2)

        @given(st_pubs)
        def test_public_key_repr_round_trip(self, pkey):
            "Test that RsaPublicKey repr can be parsed by Python."
            self.assertEqual(pkey, eval(repr(pkey)))

        @given(st_keys)
        def test_key_pair_stream_round_trip(self, keyp1):
            "Test write_to/read_from for RsaKeyPair."
            buf = FileFormats.wrapped_io()
            keyp1.write_to(buf)
            buf.seek(0)
            keyp2 = RsaKeyPair.read_from(buf)
            self.assertEqual(keyp1, keyp2)

        @given(st_keys)
        def test_key_pair_repr_round_trip(self, keyp):
            "Test that RsaKeyPair repr can be parsed by Python."
            self.assertEqual(keyp, eval(repr(keyp)))

        def test_generate_str(self):
            "Test key generation and to-string, just to cover them."
            key = RsaKeyPair.generate()
            str(key)
            str(key.pub)

    class TestCLI(unittest.TestCase):
        "Run each of the CLI commands."

        def test_setup_logging_verbose(self):
            "Verify logging setup with -v flag."
            with capture_logs() as log:
                setup_logging(verbose=1)
                self.assertIn("INFO::Logging enabled", log.getvalue())
                self.assertEqual(logging.root.level, logging.INFO)

        def test_setup_logging_extra_verbose(self):
            "Verify logging setup with -vv flag."
            with capture_logs() as log:
                setup_logging(verbose=2)
                self.assertIn("DEBUG::Logging enabled", log.getvalue())
                self.assertEqual(logging.root.level, logging.DEBUG)

        def test_run_rand(self):
            "Run 'rand BITS'"
            with capture_output() as out:
                run_rand(Namespace(bits="128"))
            self.assertIn("0x", out.getvalue())

        def test_run_rand_to_stream(self):
            "Run 'rand -o-'"
            with capture_output() as out:
                run_rand(Namespace(output=sys.stdout, bits=256))
            self.assertIn("BEGIN KEY", out.getvalue())

        def test_run_rand_to_file(self):
            "Run 'rand -o FILENAME'"
            buf = FileFormats.wrapped_io()
            buf.buffer.name = "yyyy.key"
            with capture_output("stderr") as out:
                run_rand(Namespace(output=buf, bits=256))
            self.assertIn("Saved to yyyy.key", out.getvalue())

        @given(st.binary(), TestS20.keys, st.booleans())
        def test_run_s20enc_dec(self, data, key, binary):
            "Ensure round-trip of 's20enc' with 's20dec'"
            with provide_stdin_bytes(data):
                with capture_stdout_bytes() as out:
                    run_s20enc(Namespace(key=key, binary=binary))
            with provide_stdin_bytes(out.getvalue()):
                with capture_stdout_bytes() as out:
                    run_s20dec(Namespace(key=key))
            self.assertEqual(out.getvalue(), data)

        @given(TestS20.keys, st.text(string.ascii_letters))
        def test_run_s20enc_uses_note(self, key, note):
            "Ensure that s20enc implements note header option."
            buf = FileFormats.wrapped_io()
            run_s20enc(Namespace(input=buf, key=key, note=note, output=buf))
            buf.seek(0)
            with FileFormats.reader(buf) as inp:
                self.assertEqual(inp.banner, FileFormats.S20_DATA)
                self.assertEqual(inp.headers["Note"], note)

    unittest.main(
        argv=[cli_args.prog],
        verbosity=cli_args.verbose + 1,
        # buffer=True,
        failfast=cli_args.failfast,
    )
