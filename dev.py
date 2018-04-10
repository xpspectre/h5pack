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
    # a = None
    a = [None, None, None]

    # a = {
    #     'a': 123,
    #     'b': 456,
    #     'cd': 789
    # }

    # a = {
    #     1: 123,
    #     2: 456,
    #     4: 789
    # }

    # a = {
    #     'a': 123,
    #     'b': 'cde'
    # }

    # a = {
    #     11: 123,
    #     22: 'cde'
    # }

    # a = {1, 2, 3}
    # a = {'a', 'ab', 'abc'}
    # a = {'a', 1}

    # a = {
    #     'a': 123,
    #     'b': 'abc',
    #     'c': np.ones((2,4)),
    #     1: {'qqq', 'rrr', 'sss'},
    #     2: {'x': 1.2, 'y': 3.5}
    # }

    # a = {
    #     'a': {
    #         'aa': {
    #             'aaa': 789,
    #             'bbb': 'fjaiofj0'
    #         },
    #         'bb': np.zeros((1,2))
    #     },
    #     'b': 123
    # }

    # a = [np.zeros((4,3)), np.ones((6,))]

    # a = {
    #     'a': np.zeros((4, 3)),
    #     'b': np.ones((6,)),
    #     5: np.ones((1, 2))
    # }

    # a = np.int64(123)

    a_file = os.path.join(test_dir, 'a.h5')
    pack(a, a_file)
    a_ = unpack(a_file)

    return 0


if __name__ == '__main__':
    main()
