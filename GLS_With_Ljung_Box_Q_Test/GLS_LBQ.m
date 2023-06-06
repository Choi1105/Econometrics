%% GLS under Heteroskedasticity
clear;
clc;

%% STEP .1 : DGP
T = 1000;
tau = 400;
beta1 = -3;
beta2 = 5;
sig12 = 0.1;
sig22 = 2;

ym = zeros(T,1);
xm = [ones(T,1), 5*rand(T,1)];

for t = 1:T

    if t <= tau
        ym(t,1) = beta1*xm(t,1) + beta2*xm(t,2) + randn(1,1)*sqrt(sig12); 
    else
        ym(t,1) = beta1*xm(t,1) + beta2*xm(t,2) + randn(1,1)*sqrt(sig22); 
    end

end

%% STEP .2 : Test for Heteriscadasticity
Y = ym;
X = xm;
tau = 400;

printi = 0;
[bhat_OLS, sig2hat_OLS, ~, ~, ~, ehat_OLS, varbhat_OLS, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y,X,printi);

% Primitive
plot(ehat_OLS);
title('Residuals');

% Ljung Box Q test
mLag = 24;
Rho = autocorr(ehat_OLS.^2, mLag);
Qm = zeros(mLag,1);
for i = 1:mLag
    Qm(i) = Rho(i)^2/(T-i);
end
Q_test = T*(T+2)*sum(Qm);
p_val = 1 - cdf('chi2', Q_test, mLag);

%% STEP .3 : GLS given tau
% before break
Y1 = Y(1:tau);
X1 = X(1:tau, :);
[~, sig12hat, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,X1,printi);

% before break
Y2 = Y(tau+1:T);
X2 = X(tau+1:T,:);
[~, sig22hat, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,X2,printi);

% Constucting Omega (variance-covariance matrix)
Dm = [zeros(tau,1); ones(T-tau,1)];
V = sig12hat*(1-Dm) + sig22hat*(Dm);
Omega_hat = diag(V);

% GLS Estimator
bhat_GLS = inv(X'*inv(Omega_hat)*X)*X'*inv(Omega_hat)*Y;
varbhat_GLS = inv(X'*inv(Omega_hat)*X);

%% STEP .4 : Results
disp('=====================================');
disp(['Ljung-Box Q-test  ', num2str(Q_test)]);
disp(['P value is  ', num2str(p_val)]);
disp('=====================================')
disp(['GLS estimator is  ', num2str(bhat_GLS')]);
disp(['GLS variance is  ', num2str(diag(varbhat_GLS)')]);
disp('=====================================')
disp(['OLS estimator is  ', num2str(bhat_OLS')]);
disp(['OLS variance is  ', num2str(diag(varbhat_OLS)')]);
disp('=====================================')

