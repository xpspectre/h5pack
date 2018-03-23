# h5pack - HDF5 Easy Serialization Library

Schema-less packing/unpacking of simple Python data structures to hdf5. The original use case was to pack and unpack some (potentially large) Numpy arrays and metadata w/o too much boilerplate code.

This file format should be easy to read/write w/ this library and easy to read w/o this library (using the HDF Viewer or just a bare HDF5 library).

## Usage

    pack(data, filename, **options)
    
    data = unpack(filename)

`data` is a `str`, `int`, `float`, or any of the scalar Numpy numeric types; or a Numpy `ndarray` of any Numpy numeric type; or a `tuple`, `list`, `set`, or `dict` of the above.

The *collection* types `tuple`, `list`, `set`, and `dict` may be *homogeneous* or *heterogeneous*. Homogeneous means all elements are the same type. Dicts have this repeated 2x: 1 for keys and 1 for vals. These are stored as a dataset vector for convenience and efficiency. Heterogeneous means elements have different types. These are stored with indexes/keys as nested groups and elements/vals inside them. Heterogeneous dict keys are coerced to strings on pack (and coerced back on unpack).

## Limitations

May expand the functionality; may decide not to for performance/simplicity.

- Doesn't support arbitrary objects. Could add a mechanism, possibly based on JSON's `object_hook` or `functools.singledispatch` to let a user add custom pack/unpack functions.
- Options except for compression aren't really implemented yet - TODO.