import string
import random
import os
import numpy as np
from nose.tools import assert_equal
from h5pack import pack, unpack


class TestH5Pack():
    """Test roundtripping (pack and then unpack gives the same result)
    TODO: Test actual file format when it's nailed down
    """
    def setup(self):
        """Generate random filename for export
        Could be used for parallel testing later
        """
        self.testdir = 'data'
        self.filename = os.path.join(self.testdir, ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)) + '.h5')

    def teardown(self):
        """Delete the file, if it exists"""
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def check_roundtrip(self, x, debug=False):
        """Helper method that packs, unpacks, and checks that the data x is unchanged.
        Has an extra option to make a copy of the file for debugging.
        """
        pack(x, self.filename)
        if debug:
            pack(x, self.filename + '_debug.h5')
        x_ = unpack(self.filename)
        assert_equal(x, x_)  # the simple comparison should work for most things

    def test_single_str(self):
        self.check_roundtrip('abc')

    def test_single_int(self):
        self.check_roundtrip(42)

    def test_single_float(self):
        self.check_roundtrip(1.23)

    def test_single_npnum(self):
        self.check_roundtrip(np.int64(123))
        self.check_roundtrip(np.float32(1.234))

    # Homogeneous indexed collections
    def test_list_strs(self):
        self.check_roundtrip(['abc', 'def', 'ghij'])

    def test_list_ints(self):
        self.check_roundtrip([1, 2, 3, 4])

    def test_list_floats(self):
        self.check_roundtrip([1.23, 4.56, 7.89])

    def test_tuple_strs(self):
        self.check_roundtrip(('abc', 'def', 'ghij'))

    def test_tuple_ints(self):
        self.check_roundtrip((1, 2, 3, 4))

    def test_tuple_floats(self):
        self.check_roundtrip((1.23, 4.56, 7.89))

    # Heterogeneous indexed collections
    def test_tuple_mixed(self):
        self.check_roundtrip((123, 'abc'))

    def test_list_mixed(self):
        self.check_roundtrip([123, 'abc'])

    # Nested (heterogeneous) indexed collections
    def test_nested_tuple(self):
        self.check_roundtrip(('abc', [123, 456], 1.23), debug=True)