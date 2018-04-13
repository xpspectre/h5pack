% Misc Matlab dev script
%   Read Python-generated file and output another
% Notes:
%   Python is row/C major, Matlab is col/F major. HDF5 stores data row/C major.
%       So numpy arrays (nonscalar, dim > 1 datasets)  need to be transposed if reading from Python
%   1-D Python numpy array -> Matlab row vector
%   Cell arrays of strings from homegenous string collections are padded with
%       ASCII null = char(0). TODO: Implementing stripping these.
%   Tuples and lists -> same thing in Matlab
%   Matlab doesn't have None/null. A single null is returned as an empty cell
%       array. A None/null in a collection is returned as a cell array of empty
%       cells. 
%   Matlab's containers.Map must have keys of the same type. This errors if it
%       tries to unpack a dict with heterogeneous keys. 
%     Related - sets have the same restriction - can only be homegeneous
%       entries. Workaround is to just return a cell array.
%   Heterogeneous dicts with str keys (that are valid identifiers) are made 
%       into structs. This is the standard Matlab usage (moreso than containers.Map)
%       TODO: Possibly make this an option.
%       TODO: Also make an option for homogeneous dicts to be structs as well
clear; close all; clc;

a_file = 'data/a.h5';
% a_file = 'data/net1_0.h5';

h5disp(a_file);

a = h5unpack(a_file);

% TODO: Implement packing
