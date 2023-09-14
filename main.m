clear all; close all
N = 10000;       % number of data
Nhalf = N/2;     % half data for training; half for testing
d = 50;          % dimension
rho1 = 0.9;      % parameter for covariance matrix in distribution 1
mu1 = 0;         % mean vector for distribution 1
rho2 = 0.7;      % parameter for covariance matrix in distribution 1
mu2 = 0.5;       % mean vector for distribution 2
p1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; % prior prob

% generate samples
xd1 = xGen(d, N, mu1, rho1);
xd2 = xGen(d, N, mu2, rho2);

% training phase
[xmean1, Cx1] = train(xd1(:, 1:Nhalf)) % mean, covariance of class-1
[xmean2, Cx2] = train(xd2(:, 1:Nhalf)) % mean, covariance of class-2

% discriminator
[log_detC1, Bh1] = disc(Cx1);
[log_detC2, Bh2] = disc(Cx2);

% test phase
Np = length(p1);
er1 = zeros(Np,1);
er2 = zeros(Np,1);
for i = 1: Np
    q1 = p1(i); % prior probability for class 1
    q2 = 1-q1;  % prior probability for class 2
    c1 = 2*log(q1) - log_detC1;
    c2 = 2*log(q2) - log_detC2;

    %%%%%%%%%%%%%% test
    class = 1;
    err1 = 0.;
    for k=Nhalf+1:N   % use MAP classifier for class-1
        [val, decision] = max([logMAP(xd1(:, k), c1, xmean1, Bh1) , ...
            logMAP(xd1(:, k), c2, xmean2, Bh2)]);
        if(class ~= decision)
            err1 = err1 + 1;
        end
    end
    fprintf("class-1 error rate = %f\n", err1/Nhalf); % class-1 error
    er1(i) = err1/Nhalf;

    class = 2;
    err2 = 0.;
    for k=Nhalf+1:N   % use MAP classifer for class-2
        [val, decision] = max([logMAP(xd2(:, k), c1, xmean1, Bh1) , ...
            logMAP(xd2(:, k), c2, xmean2, Bh2)]);
        if(class ~= decision)
            err2 = err2 + 1;
        end
    end
    fprintf("class-2 error rate = %f\n", err2/Nhalf); % class-2 error
    er2(i) = err2/Nhalf;
end

plot(p1, er1, 'b-', p1, er2, 'r-.'), legend('class-1', 'class-2'), ...
    xlabel('p(w_1)'), ylabel('P(error)'), title('MAP Classifier')
