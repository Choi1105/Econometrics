clear
clc
printi = 1;
% Model
% Y = X1*B1 + X2*B2 + e  e ~ iidN(0,sig2)
%% Step 1: DGP

T = 10000;
B1 = 1.2;
B2 = 5.4;

sig2 = 0.6;

X1m = ones(T,1);
X2m = 5*rand(T,1);
em = sqrt(sig2)*randn(T,1);

Ym = X1m*B1 + X2m*B2 + em;

%% Step 2-2: Estimation For Restricted Model (MLE)
Y = Ym;
X = [X1m];
T = rows(Y);
k = cols(X);

% Estimation
theta0 = [0;1];
Data = [Y X];
index = 1:2;
index = index';

[thetamx_LM, fmax_LM, V_LM, Vinv_LM] = SA_Newton(@lnlik_LM, @paramconst_LM, theta0, Data, printi, index);

diag_V_LM = diag(V_LM);
stde_LM = sqrt(diag_V_LM);
t_val_LM = thetamx_LM./stde_LM;
p_val = 2*(1 - cdf("t",abs(t_val_LM),T-k));

%% LM Test
% H0 : B2 = 0

X = [X1m X2m];
e_til = Y - X1m*thetamx_LM(1);
LM_Result = T*(e_til'*X*invpd(X'*X)*X'*e_til)/(e_til'*e_til);
P_val_LM = 2*(1 - cdf("Chisquare",abs(LM_Result),1))