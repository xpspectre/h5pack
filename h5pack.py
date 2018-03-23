import h5py
import numpy as np

numeric_types = {int, float}
primitive_types = {int, float, str, np.ndarray}  # Numpy array is treated as a single thing
collection_types = {tuple, list, dict, set}
indexed_types = {tuple, list}
associative_types = {dict, set}

collection_type_strs = {x.__name__ for x in collection_types}


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
    if x_type in primitive_types or is_number_type(x_type):
        return True
    return False


def is_collection_type(x_type):
    if x_type in collection_types:
        return True
    return False


def is_collection_str(x_str):
    if x_str in collection_type_strs:
        return True
    return False


def is_indexed_homogeneous(data):
    """Returns True for homogeneous, False for heterogeneous.
    TODO: Special case of ints and floats mixed -> homogeneous float
    """
    type0 = type(data[0])
    for item in data:
        item_type = type(item)
        if item_type != type0 or item_type in collection_types:
            return False
    return True


def is_associative_homogeneous(data):
    """Returns True for homogeneous, False for heterogeneous."""
    k0, v0 = next(iter(data.items()))
    ktype0 = type(k0)
    vtype0 = type(v0)
    if ktype0 in collection_types or vtype0 in collection_types:
        return False
    for k, v in data.items():
        if type(k) != ktype0 or type(v) != vtype0:
            return False
    return True


def validate_inds(keys):
    """Validate that the string keys in a group's sub-items form a valid set of indexes for a list/tuple. Raise
    ValueError if it fails. Note: keys strs are dumb lex order.
    """
    inds = sorted(int(ind) for ind in keys)
    target_inds = list(range(len(inds)))
    if inds != target_inds:
        raise ValueError('Keys don''t make up valid indexes')


def write_attrs(ds, attrs):
    """Write dataset attributes dict, including special handling of 'type' attr."""
    for k, v in attrs.items():
        if k == 'data_type' or k == 'collection_type':
            try:  # For Python types
                v = v.__name__
            except AttributeError:  # For Numpy types
                v = str(v)
        ds.attrs[k] = v


def write_primitive(group, name, data):
    """Note: No dataset chunk options (like compression) for scalar"""
    data_type = type(data)

    # Write dataset
    if data_type == str:
        ds = group.create_dataset(name, data=np.string_(data))
    else:
        ds = group.create_dataset(name, data=data)

    # Write attrs
    write_attrs(ds, {'data_type': data_type, 'collection_type': 'primitive'})


def read_primitive(group, name):
    """"""
    ds = group[name]
    data_type = str_type_map[ds.attrs['data_type']]
    val = ds[...]
    if data_type == str:
        val = str(np.char.decode(val, 'utf-8'))
    elif data_type == int or data_type == float:
        val = data_type(val)  # Convert back to Python number type
    elif data_type == np.ndarray or is_number_type(data_type):
        pass  # Numpy type, keep as is
    else:
        raise ValueError('Scalar data type not recognized')
    return val


def write_indexed(group, name, data, ds_kwargs):
    """
    2 cases:
        1. Homogeneous primitives: turn into a single dataset
        2. Heterogeneous
    """
    # See if this is homogeneous primitives
    data_type = type(data)
    homegeneous = is_indexed_homogeneous(data)
    type0 = type(data[0])

    # write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

    # Save homogenous index collection type as numpy array
    if homegeneous:
        item_type = type0
        if is_number_type(item_type):
            ds = group.create_dataset(name, data=data, **ds_kwargs)
        elif item_type == str:
            ds = group.create_dataset(name, data=np.string_(data),  **ds_kwargs)
        else:
            raise Exception('should not reach here')
        write_attrs(ds, {'data_type': item_type, 'collection_type': data_type, 'homogeneous': True})
    else:
        sub_group = group.create_group(name)
        for i, item in enumerate(data):
            item_type = type(item)
            ind_str = '{}'.format(i)
            if is_collection_type(item_type):
                write_collection(sub_group, ind_str, item, ds_kwargs)
            else:
                write_primitive(sub_group, ind_str, item)
        write_attrs(sub_group, {'data_type': data_type, 'collection_type': data_type, 'homogeneous': False})


def read_indexed(group, name):
    """"""
    collection_type = str_type_map[group[name].attrs['collection_type']]
    data_type = str_type_map[group[name].attrs['data_type']]
    homogeneous = bool(group[name].attrs['homogeneous'])

    # Read homogeneous array as single val
    if homogeneous:
        ds = group[name]
        item_type = str_type_map[ds.attrs['data_type']]
        vals = ds[...]
        if item_type == str:
            vals = list(val.decode('utf-8') for val in vals)
        else:
            vals = list(item_type(val) for val in vals)
    else:
        sub_group = group[name]
        keys = sub_group.keys()
        validate_inds(keys)
        vals = [None] * len(keys)
        for ind_str in sub_group.keys():
            ind = int(ind_str)
            vals[ind] = read_data(sub_group, ind_str)

    # Convert list to tuple if needed
    if collection_type == tuple:
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
        homogeneous_type = is_associative_homogeneous(data)
        write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

        # Save homogeneous dict as 2 arrays
        if homogeneous_type == 'homogeneous':
            keys = []
            vals = []
            for k, v in sorted(data.items()):
                keys.append(k)
                vals.append(v)
            ktype = type(k)
            vtype = type(v)
            if ktype == str:
                keys = np.string_(keys)
            if vtype == str:
                vals = np.string_(vals)

            ds_keys = group.create_dataset('keys', data=keys)
            ds_vals = group.create_dataset('vals', data=vals)
            write_attrs(ds_keys, {'data_type': ktype})
            write_attrs(ds_vals, {'data_type': vtype})
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
        ktype = type(k0)
        for k in data:
            if type(k) != ktype:
                homogeneous_type = 'heterogeneous'
        write_attrs(group, {'collection_type': data_type, 'homogeneous_type': homogeneous_type})

        # Save homogeneous set as an array, allowing any type
        if homogeneous_type == 'homogeneous':
            keys = list(data)
            if ktype == str:
                keys = np.string_(keys)
            ds_keys = group.create_dataset('keys', data=keys)
            write_attrs(ds_keys, {'data_type': ktype})
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


def write_collection(group, name, data, ds_kwargs):
    """"""
    data_type = type(data)

    # Check whether collection is indexed or associative
    if data_type in indexed_types:
        write_indexed(group, name, data, ds_kwargs)
    elif data_type in associative_types:
        write_associative(group, name, data, ds_kwargs)
    else:
        raise Exception('should not reach here')


def read_collection(group, name):
    """"""
    collection_type = str_type_map[group[name].attrs['collection_type']]

    if collection_type in indexed_types:
        return read_indexed(group, name)
    elif collection_type in associative_types:
        return read_associative(group, name)
    else:
        raise Exception('Collection type not recognized')


def write_data(group, name, data, ds_kwargs):
    """Main data writing function, which is called recursively. Does the heavy lifting of determining the type and
    writing the data accordingly.

    Args:
        group: Previous group this will be attached to
        name: Name of current group or dataset to hold this data
        data: Data to store
        ds_kwargs: Options
    """
    data_type = type(data)

    # Check whether type is primitive or collection
    if is_primitive_type(data_type):
        write_primitive(group, name, data)
    elif is_collection_type(data_type):
        write_collection(group, name, data, ds_kwargs)
    else:
        raise ValueError('Data not one of the valid primitive or collection types')


def read_data(group, name):
    """"""
    collection_type_str = group[name].attrs['collection_type']
    data_type = str_type_map[group[name].attrs['data_type']]

    if is_collection_str(collection_type_str):
        return read_collection(group, name)
    elif is_primitive_type(data_type):
        return read_primitive(group, name)
    else:
        raise ValueError('Data type not recognized')


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
        write_data(f, 'root', data, ds_kwargs)


def unpack(filename):
    """Unpack data from filename"""
    with h5py.File(filename, 'r') as f:
        # Recursively build up read data
        data = read_data(f, 'root')
    return data
