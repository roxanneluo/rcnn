function data = my_read(filename, mat_size)
assert(numel(mat_size) == 2);
mat_size = mat_size([2,1]);
f = fopen(filename, 'r');
data = fread(f, mat_size, 'single')';
fclose(f);
