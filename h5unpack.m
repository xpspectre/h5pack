function data = h5unpack(filename)
% Matlab API of h5pack unpack function
%   Should be able to unpack files compatibly with Python
info = h5info(filename);
% Use the full path of each element
%   Groups have full paths
%   Datasets just have their name
data = read_data(info, '/root', filename);
end

function data = read_data(group, name, filename)
dataset = find_dataset(group, name);
if is_primitive(dataset)
    data = read_primitive(dataset, name, filename);
else
    data = read_collection(dataset, name, filename);
end
end

function dataset = find_dataset(group, name)
% Search group for name of dataset/subgroup
ng = length(group.Groups);
for i = 1:ng
    group_ = group.Groups(i);
    if strcmp(group_.Name, name)
        dataset = group_;
        return
    end
end
name_parts = split(name, '/');
name = name_parts{end};
nd = length(group.Datasets);
for i = 1:nd
    dataset = group.Datasets(i);
    if strcmp(dataset.Name, name)
        return
    end
end
end

function data = read_primitive(dataset, name, filename)
data = h5read(filename, name);

data_type = get_attr(dataset, 'data_type');
switch data_type
    case 'ndarray'
        data = permute(data, [ndims(data):-1:1]); % Fix row vs col major. Transpose for 2-D, permute all dims for higher
    case 'bool'
        data = data == 1;
    case 'NoneType'
        data = {}; % Dummy single null
    otherwise % simple number or string datatype, keep as is
        
end
end

function data = read_collection(dataset, name, filename)
collection_type = get_attr(dataset, 'collection_type');
if ismember(collection_type, {'tuple', 'list'})
    data = read_indexed_collection(dataset, name, filename);
elseif ismember(collection_type, {'dict', 'set'})
    data = read_associative_collection(dataset, name, filename);
end
end

function data = read_indexed_collection(dataset, name, filename)
if is_homogeneous(dataset)
    data = h5read(filename, name);
    
    data_type = get_attr(dataset, 'data_type');
    if strcmp(data_type, 'bool')
        data = data == 1;
    end
else % group containing nonhomogeneous collection elements
    % Internals could be datasets or further groups
    % Names are numbers from '0' to whatever
    n = length(dataset.Datasets) + length(dataset.Groups);
    data = cell(n, 1);
    
    % Read groups recursively
    ng = length(dataset.Groups);
    for i = 1:ng
        group_ = dataset.Groups(i);
        name_parts = split(group_.Name, '/');
        ind_ = str2double(name_parts{end}) + 1; % Matlab uses 1-based indexing
        data{ind_} = read_data(dataset, group_.Name, filename);
    end
    
    % Read datasets
    nd = length(dataset.Datasets);
    for i = 1:nd
        dataset_ = dataset.Datasets(i);
        name_ = strcat(name, '/', dataset_.Name);
        data_ = read_primitive(dataset_, name_, filename);
        ind_ = str2double(dataset_.Name) + 1; % Matlab uses 1-based indexing
        data{ind_} = data_;
    end
end
end

function data = read_associative_collection(dataset, name, filename)
collection_type = get_attr(dataset, 'collection_type');
if is_homogeneous(dataset)
    switch collection_type
        case 'dict'
            keys = h5read(filename, strcat(name, '/', 'keys'));
            vals = h5read(filename, strcat(name, '/', 'vals'));
            % TODO: Cleanup cell array of strings ASCII null padding
            data = containers.Map(keys, vals);
        case 'set'
            vals = h5read(filename, strcat(name)); % stored directly
            % TODO: Cleanup cell array of strings ASCII null padding
            data = containers.Map(vals, zeros(length(vals), 1)); % map with dummy vals
        otherwise
            error('Homegeneous associative collection type not recognized')
    end
else
    switch collection_type
        case 'dict'
            % Internals could be datasets or further groups
            % Names are dict keys or set elements
            n = length(dataset.Datasets) + length(dataset.Groups);
            keys = cell(n, 1);
            vals = cell(n, 1);
            
            % Read groups recursively
            j = 1; % Overall index for data
            ng = length(dataset.Groups);
            for i = 1:ng
                group_ = dataset.Groups(i);
                name_parts = split(group_.Name, '/');
                key = name_parts{end};
                val = read_data(dataset, group_.Name, filename);
                
                keys{j} =  key;
                vals{j} = val;
                j = j + 1;
            end
            
            % Read datasets
            nd = length(dataset.Datasets);
            for i = 1:nd
                dataset_ = dataset.Datasets(i);
                key = dataset_.Name;
                val = read_primitive(dataset_, strcat(name, '/', key), filename);
                
                key_type = get_attr(dataset_, 'key_type');
                if is_integer_type(key_type)
                    key = str2double(key); % TODO: Exact integer type
                end
                
                keys{j} =  key;
                vals{j} = val;
                j = j + 1;
            end
            
            % Make Matlab container - struct if keys are all strings; map otherwise
            % 	Note that a map requires keys be the same type
            if iscellstr(keys)
                data = struct();
                for i = 1:length(keys)
                    data.(keys{i}) = vals{i};
                end
            else
                data = containers.Map(keys, vals, 'UniformValues', false);
            end
        case 'set'
            % This doesn't have a great Matlab analog - just return as a cell
            % array. Treat it like an indexed collection
            data = read_indexed_collection(dataset, name, filename);
        otherwise
            error('Heterogeneous associative collection type not recognized')
    end
end
end

function val = get_attr(dataset, attr)
mask = ismember({dataset.Attributes.Name}, attr);
val = dataset.Attributes(mask).Value{1};
end

function tf = is_primitive(dataset)
collection_type = get_attr(dataset, 'collection_type');
if strcmp(collection_type, 'primitive')
    tf = true;
else
    tf = false;
end
end

function tf = is_homogeneous(dataset)
homegeneous = get_attr(dataset, 'homogeneous');
if strcmp(homegeneous, 'TRUE')
    tf = true;
else
    tf = false;
end
end

function tf = is_integer_type(type_name)
if ismember(type_name, {'int','int8','int16','int32','int64','uint8','uint16','uint32','uint64'})
    tf = true;
else
    tf = false;
end
end
