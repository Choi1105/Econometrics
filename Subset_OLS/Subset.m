clear
clc

% Model
% Y = X1*B1 + X2*B2 + e  e ~ iidN(0,sig2)

%% Step 1: DGP

T = 500;
B1 = 1.2;
B2 = 5.4;

sig2 = 0.6;

X1m = ones(T,1);
X2m = 5*rand(T,1);

em = sqrt(sig2)*randn(T,1);

Ym = X1m*B1 + X2m*B2 + em;


%% Step 2: Estimation (OLS)
Y = Ym;
X = [X1m X2m];
k = 2;


bhat = inv(X'*X)*X'*Y;
Yhat = X*bhat;
ehat = Y - Yhat;
sig2hat = ehat'*ehat/(T-k);
varbhat = sig2hat*inv(X'*X);
stde = sqrt(diag(varbhat));

RSS = ehat'*ehat;
TSS = (Y - mean(Y))'*(Y - mean(Y));
R2 = 1- RSS/TSS;
R2_ = 1 - (RSS*(T-1))/(TSS*(T-k));
SC = log(RSS/T) - k*log(T)/T;
AIC = log(RSS/T) - 2*k/T;

%% Step 3: Subset of Regression
Mx1 = eye(T) - X1m*inv(X1m'*X1m)*X1m';
Mx2 = eye(T) - X2m*inv(X2m'*X2m)*X2m';

Sub_Bhat1 = inv(X1m'*Mx2*X1m)*X1m'*Mx2*Y;
Sub_Bhat2 = inv(X2m'*Mx1*X2m)*X2m'*Mx1*Y;

Sub_Bhat = [Sub_Bhat1 ; Sub_Bhat2];
True_B = [B1 ; B2];

disp([True_B bhat  Sub_Bhat])