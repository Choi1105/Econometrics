%% IV Estimator (2SLS)
clear;
clc;

%% Step.1 : DGP
% Data Info
T = 1000;

% Model Param
beta = [6 ; 4.2 ; 3.7];
sig2 = 1.9;

% Common Factor
C2 = randn(T,1)*2;
C3 = randn(T,1)*3;

% Independent X
only_X2 = randn(T,1);
only_X3 = randn(T,1);

X2m = C2 + only_X2;
X3m = C3 + only_X3;

% Correlated Error with X2 and X3
em =  C2 + C3 + randn(T,1)*sqrt(sig2);

% Instrumental Variables
Z2m = only_X2 + randn(T,1);
Z3m = only_X3 + randn(T,1);

Xm = [ones(T,1) X2m X3m];
Zm = [Xm(:,1) Z2m Z3m];

% Dependent Variable
Ym = Xm*beta + em;

%% Step.2 : Estimation
% Data
Y = Ym;
X = Xm;
Z = Zm;
T = rows(X);
k = cols(X);

% OLS
printi = 0;
[bhat_OLS, sig2hat, stde, ~, ~, ~, varbhat_OLS, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y,X,printi);
b0 = [6;1.4;3.7];  % null hypothesis
t_val = (bhat_OLS - b0)./ stde; % k by 1, t values
p_val_OLS = 2*(1-cdfn(abs(t_val))); % k by 1, t values

% 2SLS
Pz = Z*inv(Z'*Z)*Z';
XPzX = X'*Pz*X;
XPzY = X'*Pz*Y;
bhat_2SLS = inv(XPzX)*XPzY;

Yhat = X*bhat_2SLS;
e_hat = Y - Yhat;
sig2hat_2SLS = e_hat'*e_hat/(T-k); % U_hat이 아님.
varbhat_2SLS = sig2hat_2SLS*inv(XPzX);

%% Step.3 : Hausman-Wu Test
q = bhat_2SLS - bhat_OLS;
var_q = varbhat_2SLS - varbhat_OLS;

Hausman_Wu = q'*inv(var_q)*q;
p_val = 1 - cdf('chi2', Hausman_Wu, k);

%% Step.4 : Results
disp('====================');
disp(['     OLS     ', '      2SLS']);
disp([bhat_OLS bhat_2SLS]);
disp("====================")
disp(['Hausman_Wu Test Statistics   ', num2str(Hausman_Wu)])
disp(['P_value                                              ', num2str(p_val)])
disp("====================")
disp(['   OLS P_value   '])
disp([p_val_OLS])