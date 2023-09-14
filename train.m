function [xmean, Cx] = train(x)
d = size(x,1);
N = size(x,2);
xmean = zeros(d,1);
for k=1:N
    xmean = xmean + x(:, k);
end
xmean = xmean / N;

Cx = zeros(d);
for k=1:N
    x0 = x(:, k) - xmean;
    Cx = Cx + x0 * x0';    
end
Cx = Cx / N;