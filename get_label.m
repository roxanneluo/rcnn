function label = get_label(nums)
label = zeros(sum(nums),1);
num_start = 1;
for i = 1:length(nums)
  num_end = num_start+nums(i)-1;
  label(num_start:num_end) = i;
end
