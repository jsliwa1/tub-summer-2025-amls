import struct
import zipfile

def read_zip_binary(path):
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split('/')[-1].split('.')[0] 
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array , r)
    return ragged_array

def read_binary(path): 
    ragged_array = []
    with open(path , "rb")as r:
        read_binary_from(ragged_array , r)
    return ragged_array

def read_binary_from(ragged_array, r): 
    while(True):
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2))) 
        ragged_array.append(sub_array)
