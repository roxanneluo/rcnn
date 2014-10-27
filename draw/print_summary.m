% ------------------------------------------------------------------------
function print_summary(class_id, class, mean_pos, stdd, feat_name, folder, suffix)
% ------------------------------------------------------------------------
if ~exist('suffix', 'var')
  suffix = '';
end
file_name = [folder, sprintf('%s-%d %s-summary-%s', class, class_id, feat_name), suffix,'.csv']
f = fopen(file_name, 'w');
len = length(mean_pos);
print(f, 1:len);
print(f, mean_pos);
print(f, stdd);
fclose(f);

% ------------------------------------------------------------------------
function print(file, x)
% ------------------------------------------------------------------------
len = length(x);
for i=1:len
  fprintf(file, '%f, ', x(i));
end
fprintf(file, '\n');


