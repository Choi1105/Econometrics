%% Cochrane - Orcutt Iterative estiamtion Method
% y(t) = x(t)'*beta + e(t)
% e(t) = rho*e(t-1) + v(t), v(t) ~ N(0,sig2)
clear;
clc;

%% Step 0: DGP %%
T = 1000;

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

rho_iter = zeros(1,100);
sig_iter = zeros(1,100);
beta_iter = zeros(1,100);

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

% Result_Save
rho_iter(1) = 1;
sig_iter(1) = sig2hat;
beta_iter(1) = bhat;

%% Step 2-2 rho iterate 1. %%
T = 999;

rhohat = inv(ehat(1:T-1)'*ehat(1:T-1))*ehat(1:T-1)'*ehat(2:T);
Y_N1 = Y(2:T) - rhohat*Y(1:T-1);
X_N1 = X(2:T) - rhohat*X(1:T-1);

b_N1hat = inv(X_N1'*X_N1)*X_N1'*Y_N1;
Y_N1hat = X_N1*b_N1hat;
E_N1hat = Y_N1 - Y_N1hat;
sig2_N1hat = E_N1hat'*E_N1hat/(T-k);

% Result_Save
rho_iter(2) = rhohat;
sig_iter(2) = sig2hat;
beta_iter(2) = bhat;

%% Step 2-3 rho iterate 2. %%
T = 998;

rhohat2 = inv(E_N1hat(1:T-1)'*E_N1hat(1:T-1))*E_N1hat(1:T-1)'*E_N1hat(2:T);
Y_N2 = Y_N1(2:T) - rhohat2*Y_N1(1:T-1);
X_N2 = X_N1(2:T) - rhohat2*X_N1(1:T-1);

b_N2hat = inv(X_N2'*X_N2)*X_N2'*Y_N2;
Y_N2hat = X_N2*b_N2hat;
E_N2hat = Y_N2 - Y_N2hat;
sig2_N2hat = E_N2hat'*E_N2hat/(T-k);


% Result_Save
rho_iter(3) = rhohat;
sig_iter(3) = sig2_N2hat;
beta_iter(3) = b_N2hat;

%% Step 3: Inference %%
se = sqrt(diag(inv(X'*X)*sig2hat));
tstat = bhat./se;
pval = 2*(1-tcdf(abs(tstat),T-k));