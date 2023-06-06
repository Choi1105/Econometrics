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
%% Step 2-1: Estimation For Unrestricted Model(MLE)
Y = Ym;
X = [X1m X2m];
T = rows(Y);
k = cols(X);

% Estimation
theta0 = [0;0;1];
Data = [Y X];
index = 1:3;
index = index';

[thetamx, fmax, V, Vinv] = SA_Newton(@lnlik, @paramconst, theta0, Data, printi, index);

diag_V = diag(V);
stde = sqrt(diag_V);
t_val = thetamx./stde;
p_val = 2*(1 - cdf("t",abs(t_val),T-k));

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

[thetamx_LR, fmax_LR, V_LR, Vinv_LR] = SA_Newton(@lnlik_LR, @paramconst_LR, theta0, Data, printi, index);

diag_V_LR = diag(V_LR);
stde_LR = sqrt(diag_V_LR);
t_val_LR = thetamx_LR./stde_LR;
p_val_LR = 2*(1 - cdf("t",abs(t_val_LR),T-k));

%% Wald Test
% H0 : B2 = 0
LR_Result = -2*(fmax_LR - fmax)
P_val_LR = 2*(1 - cdf("Chisquare",abs(LR_Result),1))
