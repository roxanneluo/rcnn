function same = get_same(data)
n = size(data,1);
same = zeros(n);
for i = 1:n
  d = data(i,:);
  for j = i+1:n
    if all(d == data(j,:))
      same(i,j) = 1;
    end
  end
end
same = max(same, same');
