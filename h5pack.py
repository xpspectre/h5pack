import h5py
import os
import numpy as np

numeric_types = {int, float}
primitive_types = {int, float, str}
collection_types = {tuple, list, dict, set, np.ndarray}
indexed_types = {tuple, list}
associative_types = {dict, set}


def is_integer_type(x_type):
    if x_type == int or issubclass(x_type, np.integer):
        return True
    return False


def is_number_type(x_type):
    if x_type in numeric_types or issubclass(x_type, np.number):
        return True
    return False


def is_primitive_type(x_type):
    if x_type == str or is_number_type(x_type):
        return True
    return False


def write_attrs(ds, attrs):
    """Write dataset attributes dict, including special handling of 'type' attr."""
    for k, v in attrs.items():
        if k == 'data_type':
            v = v.__name__
        ds.attrs[k] = v


def write_primitive(group, data, attrs):
    """Note: No dataset chunk options (like compression) for scalar"""
    data_type = attrs['data_type']

    # Write dataset
    if data_type == str:
        ds = group.create_dataset('val', data=np.string_(data))
    elif is_number_type(data_type):
        ds = group.create_dataset('val', data=data)
    else:
        raise Exception('should not reach here')

    # Write attrs
    write_attrs(ds, attrs)


def write_indexed(group, data, attrs, ds_kwargs):
    """
    2 cases:
        1. Homogeneous primitives: turn into a single dataset
        2. Heterogeneous
    """
    # See if this is homogeneous primitives
    homogeneous_type = 'homogeneous'
    type0 = type(data[0])
    for item in data:
        item_type = type(item)
        if item_type != type0 or item_type in collection_types:
            homogeneous_type = 'heterogeneous'
            break
        # TODO: Special case of ints and floats mixed -> homogeneous float

    # Save homogenous index collection type as numpy array
    if homogeneous_type == 'homogeneous':
        item_type = type0
        if is_number_type(item_type):
            ds = group.create_dataset('val', data=data, **ds_kwargs)
        elif item_type == str:
            ds = group.create_dataset('val', data=np.string_(data),  **ds_kwargs)
        else:
            raise Exception('should not reach here')
        write_attrs(ds, attrs)
    elif homogeneous_type == 'heterogeneous':
        for i, item in enumerate(data):
            item_type = type(item)
            attrs['data_type'] = item_type
            group_i = group.create_group('{}'.format(i))  # Create new group whose name is the index
            if item_type in collection_types:
                write_collection(group_i, item, attrs, ds_kwargs)
            else:
                write_primitive(group_i, item, attrs)


def clean_key(key):
    """Ensure key is either an int or else coerced to a string"""
    if is_integer_type(type(key)):
        return key
    return str(key)


def write_associative(group, data, attrs, ds_kwargs):
    """
    Keys are either ints or strings
    """
    data_type = attrs['data_type']
    if data_type == dict:
        # See if it's homogeneous - the keys are all 1 type and he vals are all 1 type
        homogeneous_type = 'homogeneous'
        k0, v0 = next(iter(data.items()))
        ktype0 = type(k0)
        vtype0 = type(v0)
        for k, v in data.items():
            if type(k) != ktype0 or type(v) != vtype0:
                homogeneous_type = 'heterogeneous'

        # Save homogeneous dict as 2 arrays
        if homogeneous_type == 'homogeneous':
            keys = []
            vals = []
            for k, v in sorted(data.items()):
                keys.append(clean_key(k))
                vals.append(v)
            if ktype0 == str:
                keys = np.string_(keys)
            if vtype0 == str:
                vals = np.string_(vals)

            ds_keys = group.create_dataset('keys', data=keys)
            ds_vals = group.create_dataset('vals', data=vals)
        else:
            for k, v in data.items():
                val_type = type(v)
                attrs['data_type'] = val_type
                group_key = group.create_group(clean_key(k))  # Create new group whose name is key
                if val_type in collection_types:
                    write_collection(group_key, v, attrs, ds_kwargs)
                else:
                    write_primitive(group_key, v, attrs)

    elif data_type == set:
        # See if it's homogeneous - the keys are all type
        homogeneous_type = 'homogeneous'
        k0 = next(iter(data))
        ktype0 = type(k0)
        for k in data:
            if type(k) != ktype0:
                homogeneous_type = 'heterogeneous'

        # Save homogeneous set as an array
        if homogeneous_type == 'homogeneous':
            keys = []
            for k in sorted(data):
                keys.append(clean_key(k))
            if ktype0 == str:
                keys = np.string_(keys)

            ds_keys = group.create_dataset('keys', data=keys)
        else:  # Save heterogeneous sets like dicts whose val is 0
            for k in sorted(data):
                attrs['data_type'] = int
                group_key = group.create_group(clean_key(k))  # Create new group whose name is key
                write_primitive(group_key, 0, attrs)

    else:
        raise Exception('should not reach here')


def write_collection(group, data, attrs, ds_kwargs):
    """"""
    data_type = attrs['data_type']

    # Check whether collection is indexed or associative
    if data_type in indexed_types:
        write_indexed(group, data, attrs, ds_kwargs)
    elif data_type in associative_types:
        write_associative(group, data, attrs, ds_kwargs)
    elif data_type is np.ndarray:
        ds = group.create_dataset('val', data=data,  **ds_kwargs)
    else:
        raise Exception('should not reach here')


def write_data(group, data, ds_kwargs):
    """Main data writing function, which is called recursively. Does the heavy lifting of determining the type and
    writing the data accordingly.

    Args:
        group:
        data:
    """
    data_type = type(data)
    attrs = {'data_type': data_type}  # store accumulated attrs

    # Check whether type is primitive or collection
    if is_primitive_type(data_type):
        attrs['combo_type'] = 'primitive'
        write_primitive(group, data, attrs)
    elif data_type in collection_types:
        attrs['combo_type'] = 'collection'
        write_collection(group, data, attrs, ds_kwargs)
    else:
        raise ValueError('Data not one of the valid primitive or collection types')


def pack(data, filename, compression=True):
    """Pack data into filename.

    Args:
        data: str, number (int or float), ndarray, or tuple, list, dict, set of them to save
        filename: str, name of file to save
        compression: bool, whether to gzip each dataset
    """
    # Setup dataset keyword args
    ds_kwargs = {}
    if compression:
        ds_kwargs['compression'] = 'gzip'

    # Open data file
    with h5py.File(filename, 'w') as f:
        # Recursively write out data
        write_data(f, data, ds_kwargs)


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
    b_file = os.path.join(test_dir, 'b.h5')
    pack(b, b_file)

    # Tuple of mixed stuff
    c = (123, 'abcd')
    c_file = os.path.join(test_dir, 'c.h5')
    pack(c, c_file)

    # Dict of homogenous stuff - k,v pairs
    d = {
        'a': 123,
        'b': 456,
        'cd': 789
    }
    d_file = os.path.join(test_dir, 'd.h5')
    pack(d, d_file)

    # Dict of homogeneous keys and heterogenous vals
    e = {
        'a': 123,
        'b': 'cde'
    }
    e_file = os.path.join(test_dir, 'e.h5')
    pack(e, e_file)

    # Write a numpy array
    f = np.zeros((5,3))
    f_file = os.path.join(test_dir, 'f.h5')
    pack(f, f_file)

    # Write a dict with mixed stuff, including numpy arrays
    g = {
        'a': 123,
        'b': 'abc',
        'c': np.ones((2,4)),
        'd': {'qqq', 'rrr', 'sss'}
    }
    g_file = os.path.join(test_dir, 'g.h5')
    pack(g, g_file)

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

    return 0


if __name__ == '__main__':
    main()
