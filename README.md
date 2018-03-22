# h5pack - HDF5 Easy Serialization Library

Schema-less packing/unpacking of simple Python data structures to hdf5.

## Usage

    pack(data, filename, **options)
    
    data = unpack(filename)

## Limitationss

May expand the functionality; may decide not to for performance/simplicity.

- Dict keys and set elements must be `int` or `str` (anything not an integer type is coerced to string). In principle, this isn't necessary - anything that's immutable+hashable is allowed. This would include tuples, which could be treated as heterogeneous entries in the keys.
