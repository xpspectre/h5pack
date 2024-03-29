import unittest
import tempfile
import os
import numpy as np
from h5pack import pack, unpack


class TestH5Pack(unittest.TestCase):
    """Test roundtripping (pack and then unpack gives the same result)
    TODO: Test actual file format when it's nailed down
    """
    def run(self, result=None):
        # Cool way to use context managers in unit test setup/teardown: https://stackoverflow.com/questions/8416208/in-python-is-there-a-good-idiom-for-using-context-managers-in-setup-teardown
        with tempfile.TemporaryDirectory() as tempdir:
            self.filename = os.path.join(tempdir, 'tempfile')
            super().run(result)

    def check_roundtrip(self, x, debug=False):
        """Helper method that packs, unpacks, and checks that the data x is unchanged.
        Has an extra option to make a copy of the file for debugging.
        TODO: Need to recursively check elements, including calling all() on Numpy arrays?
        """
        pack(x, self.filename)
        if debug:
            pack(x, self.filename + '_debug.h5')
        x_ = unpack(self.filename)
        self.assertEqual(x, x_)  # the simple comparison should work for most things

    def check_roundtrip_ndarrays(self, x):
        """Hacked function to check whether a file w/ 1 'level' of Numpy arrays roundtrips.
        1 level is a single numpy array or 1 collection of them.
        """
        pack(x, self.filename)
        x_ = unpack(self.filename)

        xtype = type(x)
        if xtype == np.ndarray:
            np.testing.assert_array_equal(x, x_)
        elif xtype in {list, tuple}:
            for xi, xi_ in zip(x, x_):
                np.testing.assert_array_equal(xi, xi_)
        elif xtype == set:
            raise ValueError('should not reach here; can''t use ndarrays as set vals')
        elif xtype == dict:
            # Get consistent order for keys
            for k, v in x.items():
                v_ = x[k]  # The unpacked dict must have this key
                np.testing.assert_array_equal(v, v_)
        else:
            raise ValueError('should not reach here; called with wrong inputs')

    def test_single_str(self):
        self.check_roundtrip('abc')

    def test_single_int(self):
        self.check_roundtrip(42)

    def test_single_float(self):
        self.check_roundtrip(1.23)

    def test_single_bool(self):
        self.check_roundtrip(True)
        self.check_roundtrip(False)

    def test_single_none(self):
        self.check_roundtrip(None)

    def test_single_npnum(self):
        self.check_roundtrip(np.int64(123))
        self.check_roundtrip(np.float32(1.234))

    def test_ndarray(self):
        """Numpy array is a "primitive"/"scalar"""
        self.check_roundtrip_ndarrays(np.zeros((5,3)))
        self.check_roundtrip_ndarrays(np.ones((1, 3), dtype=np.int64))

    # Homogeneous indexed collections
    def test_list_strs(self):
        self.check_roundtrip(['abc', 'def', 'ghij'])

    def test_list_bools(self):
        self.check_roundtrip([True, False, True])

    def test_list_nones(self):
        self.check_roundtrip([None, None, None])

    def test_list_ints(self):
        self.check_roundtrip([1, 2, 3, 4])

    def test_list_floats(self):
        self.check_roundtrip([1.23, 4.56, 7.89])

    def test_tuple_strs(self):
        self.check_roundtrip(('abc', 'def', 'ghij'))

    def test_tuple_bools(self):
        self.check_roundtrip((True, False, True))

    def test_tuple_nones(self):
        self.check_roundtrip((None, None, None))

    def test_tuple_ints(self):
        self.check_roundtrip((1, 2, 3, 4))

    def test_tuple_floats(self):
        self.check_roundtrip((1.23, 4.56, 7.89))

    # Heterogeneous indexed collections
    def test_tuple_mixed(self):
        self.check_roundtrip((123, 'abc'))

    def test_list_mixed(self):
        self.check_roundtrip([123, 'abc'])

    def test_tuple_empty(self):
        self.check_roundtrip(())

    def test_list_empty(self):
        self.check_roundtrip([])

    # Nested (heterogeneous) indexed collections
    def test_nested_tuple(self):
        self.check_roundtrip(('abc', [123, 456], 1.23))

    # Homogeneous dicts
    def test_dict_str_int(self):
        self.check_roundtrip({'a': 123, 'b': 456, 'cd': 789})

    def test_dict_int_str(self):
        self.check_roundtrip({123: 'b', 456: 'cd', 789: 'efg'})

    def test_dict_empty(self):
        self.check_roundtrip({})

    # Heterogeneous dicts
    def test_dict_mixed_str_key(self):
        self.check_roundtrip({'a': 123, 'b': 'cde'})

    def test_dict_mixed_int_key(self):
        self.check_roundtrip({12: 123, 34: 'cde'})

    def test_dict_mixed_homogeneous(self):
        self.check_roundtrip({
            'a': [1, 2, 3],
            'b': [True, True, False]
        })

    # Homogeneous sets
    def test_set_int(self):
        self.check_roundtrip({1, 2, 3})

    def test_set_str(self):
        self.check_roundtrip({'a', 'ab', 'abc'})

    def test_set_empty(self):
        self.check_roundtrip(set())

    # Heterogeneous sets
    def test_set_mixed(self):
        self.check_roundtrip({1, 'a'})

    # Mixed everything
    def test_nested_list_dict(self):
        self.check_roundtrip([{'x': 1.2, 'y': 3.5}, 1])

    def test_mixed_everything(self):
        x = {
            'a': 123,
            'b': 'abc',
            'c': [1,2,3],
            'd': (1, 'a', 3.5),
            1: {'qqq', 'rrr', 'sss'},
            2: {'x': 1.2, 'y': 3.5},
            3: True,
            4: [True, False],
            5: None
        }
        self.check_roundtrip(x)

    # Mixed numpy arrays
    def test_mixed_ndarrays(self):
        x = [np.zeros((4,3)), np.ones((6,))]
        self.check_roundtrip_ndarrays(x)

    def test_mixed_ndarrays_dict(self):
        x = {
            'a': np.zeros((4,3)),
            'b': np.ones((6,))
        }
        self.check_roundtrip_ndarrays(x)

    def test_mixed_mixed_ndarrays_dict(self):
        x = {
            'a': np.zeros((4,3)),
            'b': np.ones((6,)),
            5: np.ones((1,2))
        }
        self.check_roundtrip_ndarrays(x)
