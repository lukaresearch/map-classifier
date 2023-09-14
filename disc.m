function [log_detC, Bh] = disc(C)

[U,D] = eig(C);
d = diag(D);
sqrt_d = sqrt(d);
log_detC = log(prod(d));
Bh = diag(1 ./ sqrt_d) * U';