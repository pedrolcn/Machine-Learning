A = magic(40000);

N = 10;
ET = zeros(1,N)';

for i=1:N
    tic
    A.^2;
    ET(i) = toc;
end

mean(ET)