function test_3d_mul(m,n,num)
A = rand(m,1,num);
B = rand(1,n,num);
tic
C = mul_3d(A, B);
toc
end
