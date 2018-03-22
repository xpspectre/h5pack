import h5py
import os


def pack(data, filename):
    """Pack data into filename"""

    return 0


def unpack(filename):
    """Unpack data from filename"""


def main():
    # Make some test data
    test_dir = 'data'

    # Single simple string
    a = 'abc'
    a_file = os.path.join(test_dir, 'a.h5')
    pack(a, a_file)

    # List of all strings
    b = ['abc', 'def', 'ghij']

    # Tuple of mixed stuff
    c = (123, 'abcd')

    # Dict of homogenous stuff - k,v pairs
    d = {
        'a': 123,
        'b': 456,
        'cd': 789
    }

    # Dict of homogeneous keys and heterogenous vals
    e = {
        'a': 123,
        'b': 'cde'
    }

    return 0


if __name__ == '__main__':
    main()
