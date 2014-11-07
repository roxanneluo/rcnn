function filter_start = get_filter_start(trans)
  filter_start = zeros(length(trans),1);
  filter_start(1) = 1;
  for i = 2:length(trans)
    filter_start(i) = filter_start(i-1) + size(trans{i-1},2);
  end
end
