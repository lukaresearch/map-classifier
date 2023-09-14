% 資工碩一 7110056029 吳明儒
function xd = xGen(dim, N, mu, rho)
r = rho.^(linspace(0,dim-1,dim));
C = toeplitz(r); % 共變異數矩陣C: stationary random process

[U,D] = eig(C); % 將共變異數矩陣進行eigendecomposition
A = U * diag(sqrt(diag(D))); % A = U*D^0.5

xd = zeros(dim, N);
for k=1:N             
    y = randn(dim,1); % y ~ N(0, I), 產生一標準常態分布資料
    x0 = A * y;       % 經A矩陣轉換使其共變異數矩陣為C
    x = x0 + mu;      % 將y ~ N(0, I) 轉換至 x ~ N(mu, C)
    xd(:, k) = x;
end