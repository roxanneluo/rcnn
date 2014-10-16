function C = mul_3d(A, B)
num = size(A,3);
assert(num == size(B,3));
C = zeros(size(A,1), size(B,2), num);
parfor i=1:num
  C(:,:,i) = A(:,:,i) * B(:,:,i);
end
