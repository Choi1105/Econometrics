%% Cochrane - Orcutt Iterative estiamtion Method
% y(t) = x(t)'*beta + e(t)
% e(t) = rho*e(t-1) + v(t), v(t) ~ N(0,sig2)
clear;
clc;

%% Step 0: DGP %%
T = 10000;

beta = 3.8;
sig2 = 56.0;
rho = 0.543;

Ym = zeros(T,1);
X1m = 2*rand(T,1);
em(1) = (sqrt(sig2)*randn(1,1))';

for t = 2:T
    em(t,1) = rho*em(t-1,1) + sqrt(sig2)*randn(1,1);
end

for t = 1:T
    Ym(t) = X1m(t)*beta +  + em(t);
end

%% Step 2: Estimation %%
Y = Ym;
X = X1m;
k = 1;

% Step 2-1: OLS Estimation %
bhat = inv(X'*X)*X'*Y;
Yhat = X*bhat;
ehat = Y - Yhat;
sig2hat = ehat'*ehat/(T-k);
sig2_old = sig2hat;
%% Ljung Box Q-test %%
%residuals = ehat;
%h = lbqtest(residuals);
%[h, pValue, stat, cValue] = lbqtest(residuals);



%% Step 2-2: Iterate until convergence %%
maxiter = 1000;
iter = 0;
rho_iter = zeros(1,maxiter);
sig_iter = zeros(1,maxiter);
beta_iter = zeros(1,maxiter);
pval_iter = zeros(1,maxiter);
%%
while iter < maxiter
    T = 10000 - iter;
    iter = iter+1;
    % Step 2-2a: Estimate rho
    rhohat = inv(ehat(1:T-1)'*ehat(1:T-1))*ehat(1:T-1)'*ehat(2:T);
    rho_iter(iter) = rhohat; 

    % calculate loss function
    L(iter) = sum(((Y-rhohat*Yhat-bhat*X)/(sqrt(1-rhohat^2))).^2);

    % Step 2-2b: Transform data
    Ym = Y(2:T) - rhohat*Y(1:T-1);
    Xm = X(2:T) - rhohat*X(1:T-1);

    % Step 2-2c: Estimate coefficients
    bhat = inv(Xm'*Xm)*Xm'*Ym;

    beta_iter(iter) = bhat;

    Yhat = X*bhat;
    ehat = Y - Yhat;
    sig2hat = ehat'*ehat/(T-k);

    sig_iter(iter) = sig2hat;

    se = sqrt(diag(inv(Xm'*Xm)*sig2hat));
    tstat = bhat./se;
    pval = 2*(1-tcdf(abs(tstat),T-k));
    pval_iter(iter) = pval;
end
%%
% Step 3: Inference
se = sqrt(diag(inv(X'*X)*sig2hat));
tstat = bhat./se;
pval = 2*(1-tcdf(abs(tstat),T-k));
