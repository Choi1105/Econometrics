clear
clc

% Model
% Y = X1*B1 + X2*B2 + X3*B3 + e  e ~ iidN(0,sig2)

%% Step 1: DGP

T = 100000;
B1 = 1.2;
B2 = 5.4;
B3 = 1.4;
sig2 = 0.6;

X1m = ones(T,1);
X2m = 5*rand(T,1);
X3m = 5*rand(T,1);
X4m = 0.01*rand(T,1);
em = sqrt(sig2)*randn(T,1);

Ym = X1m*B1 + X2m*B2 + X3m*B3 + em;


%% Step 2: Estimation (OLS)
Y = Ym;
X = [X1m X2m X3m X4m];
k = 4;

True_b = [B1 ; B2 ; B3 ];
bhat = inv(X'*X)*X'*Y;
Yhat = X*bhat;
ehat = Y - Yhat;
sig2hat = ehat'*ehat/(T-k);
varbhat = sig2hat*inv(X'*X);
stde = sqrt(diag(varbhat));
B0 = zeros(k,1);
t_val = (bhat - B0)./stde;
p_val = 2*(1 - cdf("t",abs(t_val),T-k));

RSS = ehat'*ehat;
TSS = (Y - mean(Y))'*(Y - mean(Y));
R2 = 1- RSS/TSS;
R2_ = 1 - (RSS*(T-1))/(TSS*(T-k));
SC = log(RSS/T) - k*log(T)/T;
AIC = log(RSS/T) - 2*k/T;

disp("True B")
disp([True_b])
disp("Bhat : X4m -> Useless Variable")
disp([bhat])
disp("P_Value")
disp([p_val])
