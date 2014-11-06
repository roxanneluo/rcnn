function resize_feature(feat_name, imdb_name)
file_name = [feat_name '.txt']
f = fopen(file_name);

dir = sprintf('feat_cache/%s/%s/', feat_name, imdb_name);
line = fgets(f);
while ischar(line)
  line = line(1:end-1);
  feat_file_name = [dir line];
  fprintf('%s\n', feat_file_name);
  d = load(feat_file_name);
  num_box = size(d.boxes,1);
  assert(size(d.feat,1) == num_box);
  IX = 1:num_box;
  gt = resize(d.gt, IX);
  overlap = resize(d.overlap, IX);
  class = resize(d.class,IX);
  feat = d.feat;
  boxes = d.boxes;
  save(feat_file_name, 'gt', 'overlap', 'boxes',...
      'feat', 'class');
  line = fgets(f);
end

fclose(f);

function data = resize(data, IX)
num = max(IX);
if size(data,1) ~= num
  data = data(IX,:);
end
