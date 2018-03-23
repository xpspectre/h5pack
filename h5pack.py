import h5py
import os
import numpy as np

numeric_types = {int, float}
primitive_types = {int, float, str}
collection_types = {tuple, list, dict, set, np.ndarray}
indexed_types = {tuple, list}
associative_types = {dict, set}


# For converting data_type metadata
str_type_map = {
    'int': int,
    'float': float,
    'str': str,
    'tuple': tuple,
    'list': list,
    'dict': dict,
    'set': set,
    'ndarray': np.ndarray,
    'int8': np.int8,  # Yeah, manually coding all the supported Numpy types here... could use np.ScalarType
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}


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
        if k == 'data_type' or k == 'collection_type':
            try:  # For Python types
                v = v.__name__
            except AttributeError:  # For Numpy types
                v = str(v)
        ds.attrs[k] = v


def write_primitive(group, data):
    """Note: No dataset chunk options (like compression) for scalar"""
    data_type = type(data)

    # Write dataset
    if data_type == str:
        ds = group.create_dataset('val', data=np.string_(data))
    elif is_number_type(data_type):
        ds = group.create_dataset('val', data=data)
    else:
        raise Exception('should not reach here')

    # Write attrs
    write_attrs(ds, {'data_type': data_type})


def read_primitive(group):
    """"""
    ds = group['val']
    data_type = str_type_map[ds.attrs['data_type']]
    val = ds[...]
    if data_type == str:
        val = str(np.char.decode(val, 'utf-8'))
    elif is_number_type(data_type):
        val = data_type(val)
    else:
        raise ValueError('Scalar data type not recognized')
    return val


def write_indexed(group, data, ds_kwargs):
    """
    2 cases:
        1. Homogeneous primitives: turn into a single dataset
        2. Heterogeneous
    """
    # See if this is homogeneous primitives
    data_type = type(data)

    homogeneous_type = 'homogeneous'
    type0 = type(data[0])
    for item in data:
        item_type = type(item)
        if item_type != type0 or item_type in collection_types:
            homogeneous_type = 'heterogeneous'
            break
        # TODO: Special case of ints and floats mixed -> homogeneous float
    write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

    # Save homogenous index collection type as numpy array
    if homogeneous_type == 'homogeneous':
        item_type = type0
        if is_number_type(item_type):
            ds = group.create_dataset('vals', data=data, **ds_kwargs)
        elif item_type == str:
            ds = group.create_dataset('vals', data=np.string_(data),  **ds_kwargs)
        else:
            raise Exception('should not reach here')
        write_attrs(ds, {'data_type': item_type})
    elif homogeneous_type == 'heterogeneous':
        for i, item in enumerate(data):
            item_type = type(item)
            group_i = group.create_group('{}'.format(i))  # Create new group whose name is the index
            # write_attrs(group_i, {'data_type': item_type})  # should be the index's type; redundant with the type inside the item
            if item_type in collection_types:
                write_collection(group_i, item, ds_kwargs)
            else:
                write_primitive(group_i, item)


def validate_inds(keys):
    """Validate that the string keys in a group's sub-items form a valid set of indexes for a list/tuple. Raise
    ValueError if it fails. Note: keys strs are dumb lex order.
    """
    inds = sorted(int(ind) for ind in keys)
    target_inds = list(range(len(inds)))
    if inds != target_inds:
        raise ValueError('Keys don''t make up valid indexes')


def read_indexed(group):
    """Python datatypes (not Numpy)"""
    data_type = str_type_map[group.attrs['data_type']]
    homogeneous_type = group.attrs['homogeneous_type']

    # Read homogeneous array as single val
    if homogeneous_type == 'homogeneous':
        ds = group['vals']
        item_type = str_type_map[ds.attrs['data_type']]
        vals = ds[...]
        if item_type == str:
            vals = list(val.decode('utf-8') for val in vals)
        else:
            vals = list(item_type(val) for val in vals)
    elif homogeneous_type == 'heterogeneous':
        # Make sure the indexes are valid
        keys = group.keys()
        validate_inds(keys)
        vals = [None] * len(keys)
        for ind_str, val in group.items():
            ind = int(ind_str)
            vals[ind] = read_data(val)
    else:
        raise ValueError('Homogeneous type not recognized')

    # Convert list to tuple if needed
    if data_type == tuple:
        vals = tuple(vals)

    return vals


def clean_key(key):
    """Ensure key is either an int or else coerced to a string"""
    # TODO: Possibly more sophisticated handling of this. Right now, just coerce everything into a str (when unpacking, the type may allow going back)
    if is_integer_type(type(key)):
        return str(key)
    return str(key)


def write_associative(group, data, ds_kwargs):
    """
    Keys are either ints or strings
    """
    data_type = type(data)

    if data_type == dict:
        # See if it's homogeneous - the keys are all 1 type and he vals are all 1 type
        homogeneous_type = 'homogeneous'
        k0, v0 = next(iter(data.items()))
        ktype0 = type(k0)
        vtype0 = type(v0)
        for k, v in data.items():
            if type(k) != ktype0 or type(v) != vtype0:
                homogeneous_type = 'heterogeneous'
        write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

        # Save homogeneous dict as 2 arrays
        if homogeneous_type == 'homogeneous':
            keys = []
            vals = []
            for k, v in sorted(data.items()):
                keys.append(k)
                vals.append(v)
            if ktype0 == str:
                keys = np.string_(keys)
            if vtype0 == str:
                vals = np.string_(vals)

            ds_keys = group.create_dataset('keys', data=keys)
            ds_vals = group.create_dataset('vals', data=vals)
            write_attrs(ds_keys, {'data_type': ktype0})
            write_attrs(ds_vals, {'data_type': vtype0})
        else:
            for k, v in data.items():
                ktype = type(k)
                vtype = type(v)
                group_key = group.create_group(clean_key(k))  # Create new group whose name is key
                write_attrs(group_key, {'data_type': ktype})  # Group gets val's type (int or str)
                if vtype in collection_types:
                    write_collection(group_key, v, ds_kwargs)
                else:
                    write_primitive(group_key, v)

    elif data_type == set:
        # See if it's homogeneous - the keys are all type
        homogeneous_type = 'homogeneous'
        k0 = next(iter(data))
        ktype0 = type(k0)
        for k in data:
            if type(k) != ktype0:
                homogeneous_type = 'heterogeneous'
        write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

        # Save homogeneous set as an array, allowing any type
        if homogeneous_type == 'homogeneous':
            keys = list(data)
            if ktype0 == str:
                keys = np.string_(keys)
            ds_keys = group.create_dataset('keys', data=keys)
            write_attrs(ds_keys, {'data_type': ktype0})
        else:  # Save heterogeneous sets like dicts whose val is 0, forcing keys to strings
            for k in data:  # Note: maybe sort these and disallow different types
                ktype = type(k)
                group_key = group.create_group(clean_key(k))  # Create new group whose name is key
                write_attrs(group_key, {'data_type': ktype})
                write_primitive(group_key, 0)

    else:
        raise Exception('should not reach here')


def read_associative(group):
    """Note that reading dicts and sets are more similar than writing them - possibly combine the code"""
    collection_type = str_type_map[group.attrs['collection_type']]
    homogeneous_type = group.attrs['homogeneous_type']

    if collection_type == dict:
        if homogeneous_type == 'homogeneous':
            ds_keys = group['keys']
            ktype = str_type_map[ds_keys.attrs['data_type']]
            keys = ds_keys[...]
            if ktype == str:
                keys = list(key.decode('utf-8') for key in keys)
            else:
                keys = list(ktype(key) for key in keys)

            ds_vals = group['vals']
            vtype = str_type_map[ds_vals.attrs['data_type']]
            vals = ds_vals[...]
            if vtype == str:
                vals = list(val.decode('utf-8') for val in vals)
            else:
                vals = list(vtype(val) for val in vals)

            return {k: v for k, v in zip(keys, vals)}
        else:
            d = {}
            for key, key_group in group.items():
                ktype = str_type_map[key_group.attrs['data_type']]
                if ktype != str:  # Try to turn non-str key back into original type - should just be ints
                    key = ktype(key)
                val = read_data(key_group)
                d[key] = val
            return d

    elif collection_type == set:
        if homogeneous_type == 'homogeneous':
            ds_keys = group['keys']
            ktype = str_type_map[ds_keys.attrs['data_type']]
            keys = ds_keys[...]
            if ktype == str:
                keys = list(key.decode('utf-8') for key in keys)
            else:
                keys = list(ktype(key) for key in keys)
            return set(keys)
        else:
            d = set()
            for key, key_group in group.items():
                ktype = str_type_map[key_group.attrs['data_type']]
                if ktype != str:  # Try to turn non-str key back into original type - should just be ints
                    key = ktype(key)
                d.add(key)
            return d

    else:
        raise ValueError('Associative type not recognized')


def write_collection(group, data, ds_kwargs):
    """"""
    data_type = type(data)

    # Check whether collection is indexed or associative
    if data_type in indexed_types:
        write_indexed(group, data, ds_kwargs)
    elif data_type in associative_types:
        write_associative(group, data, ds_kwargs)
    elif data_type == np.ndarray:
        ds = group.create_dataset('vals', data=data, **ds_kwargs)
        write_attrs(ds, {'data_type': data.dtype})
        write_attrs(group, {'collection_type': data_type})
    else:
        raise Exception('should not reach here')


def read_collection(group):
    """"""
    data_type = str_type_map[group.attrs['collection_type']]

    if data_type in indexed_types:
        return read_indexed(group)
    elif data_type in associative_types:
        return read_associative(group)
    elif data_type == np.ndarray:
        return group['vals'][...]
    else:
        raise Exception('Collection type not recognized')


def write_data(group, data, ds_kwargs):
    """Main data writing function, which is called recursively. Does the heavy lifting of determining the type and
    writing the data accordingly.

    Args:
        group:
        data:
    """
    data_type = type(data)

    # Check whether type is primitive or collection
    if is_primitive_type(data_type):
        write_primitive(group, data)
    elif data_type in collection_types:
        write_collection(group, data, ds_kwargs)
        write_attrs(group, {'collection_type': data_type, 'data_type': data_type})
    else:
        raise ValueError('Data not one of the valid primitive or collection types')


def read_data(group):
    """"""
    data_type_str = group.attrs.get('data_type')
    if data_type_str is None:  # primitive type not in a collection doesn't a data_type on the wrapper
        return read_primitive(group)

    collection_type_str = group.attrs.get('collection_type')
    if collection_type_str is not None:
        return read_collection(group)

    data_type = str_type_map[data_type_str]
    if data_type in collection_types:
        return read_collection(group)
    else:  # primitive type in a collection
        return read_primitive(group)


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
    with h5py.File(filename, 'r') as f:
        # Recursively build up read data
        data = read_data(f)
    return data


def main():
    # Make some test data
    test_dir = 'data'

    # Single simple string
    a = 'abc'
    a_file = os.path.join(test_dir, 'a.h5')
    pack(a, a_file)
    a_ = unpack(a_file)

    # List of all strings
    b = ['abc', 'def', 'ghij']
    b_file = os.path.join(test_dir, 'b.h5')
    pack(b, b_file)
    b_ = unpack(b_file)

    # List of numbers
    b2 = [1, 3, 5]
    b2_file = os.path.join(test_dir, 'b2.h5')
    pack(b2, b2_file)
    b2_ = unpack(b2_file)

    # Tuple of mixed stuff
    c = (123, 'abcd')
    c_file = os.path.join(test_dir, 'c.h5')
    pack(c, c_file)
    c_ = unpack(c_file)

    # Dict of homogeneous stuff - k,v pairs
    d = {
        'a': 123,
        'b': 456,
        'cd': 789
    }
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
