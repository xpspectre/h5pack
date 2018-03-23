import os
import numpy as np
from h5pack import pack, unpack


def main():
    # Make some test data
    test_dir = 'data'

    # a = 'abc'
    # a = 123
    # a = np.zeros((5,3))
    # a = ['abc', 'def', 'ghij']
    # a = [1.23, 4.56, 7.89]
    # a =('abc', 'def', 'ghij')
    # a = (123, 'abc')
    # a = ('abc', [123, 456], 1.23)

    a = {
        'a': 123,
        'b': 456,
        'cd': 789
    }

    a_file = os.path.join(test_dir, 'a.h5')
    pack(a, a_file)
    a_ = unpack(a_file)

    return

    d_file = os.path.join(test_dir, 'd.h5')
    pack(d, d_file)
    d_ = unpack(d_file)

    # Dict of homogeneous stuff - keys are not strs
    d2 = {
        1: 123,
        2: 456,
        4: 789
    }
    d2_file = os.path.join(test_dir, 'd2.h5')
    pack(d2, d2_file)
    d2_ = unpack(d2_file)

    # Dict of homogeneous keys and heterogenous vals
    e = {
        'a': 123,
        'b': 'cde'
    }
    e_file = os.path.join(test_dir, 'e.h5')
    pack(e, e_file)
    e_ = unpack(e_file)

    # Dict of homogeneous non-string keys and heterogenous vals - the keys get coerced to strings when packed and converted back when unpacked
    e2 = {
        11: 123,
        22: 'cde'
    }
    e2_file = os.path.join(test_dir, 'e2.h5')
    pack(e2, e2_file)
    e2_ = unpack(e2_file)

    # Write a numpy array
    f = np.zeros((5,3))
    f_file = os.path.join(test_dir, 'f.h5')
    pack(f, f_file)
    f_ = unpack(f_file)

    # Write a dict with mixed stuff, including numpy arrays
    g = {
        'a': 123,
        'b': 'abc',
        'c': np.ones((2,4)),
        1: {'qqq', 'rrr', 'sss'},  # heterogeneous keys
        2: {'x': 1.2, 'y': 3.5}
    }
    g_file = os.path.join(test_dir, 'g.h5')
    pack(g, g_file)
    g_ = unpack(g_file)

    # Write a nested dict of dicts - everything heterogenous
    h = {
        'a': {
            'aa': {
                'aaa': 789,
                'bbb': 'fjaiofj0'
            },
            'bb': np.zeros((1,2))
        },
        'b': 123
    }
    h_file = os.path.join(test_dir, 'h.h5')
    pack(h, h_file)
    h_ = unpack(h_file)

    # A numpy scalar
    i = np.int64(123)
    i_file = os.path.join(test_dir, 'i.h5')
    pack(i, i_file)
    i_ = unpack(i_file)

    # A heterogeneous set - weird but allowed in Python
    j = {'abc', 123}
    j_file = os.path.join(test_dir, 'j.h5')
    pack(j, j_file)
    j_ = unpack(j_file)

    # A homogeneous set of non-strings
    k = {12, 123, 1234}
    k_file = os.path.join(test_dir, 'k.h5')
    pack(k, k_file)
    k_ = unpack(k_file)

    return 0


if __name__ == '__main__':
    main()
