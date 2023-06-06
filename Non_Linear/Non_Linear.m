clear
clc
printi = 1;
% Model
% Y_t = (1-rho)(B1+X_t*B2)+rho*Y_t-1 + e_t  e_t ~ iidN(0,sig2)
% Y_t = (1-rho)*B1 + (1-rho)*X_t*B2 + rho*Y_t-1 + e_t  e_t ~ iidN(0,sig2)

%% Step 1: DGP

T = 9100;
B1 = 1.0;
B2 = 2.0;

rho = 0.5;

sig2 = 0.6;

Xm = 5*rand(T,1);
em = sqrt(sig2)*randn(T,1);

Ym = zeros(T,1);

Ym(1) = (1-rho)*B1 + (1-rho)*Xm(1)*B2  + em(1);

for t = 2:T
    Ym(t) = (1-rho)*B1 + (1-rho)*Xm(t)*B2 + rho*Ym(t-1) + em(t);
end

Ym = Ym(101:end);
Xm = Xm(101:end);
YmT = Ym(1:end-1);

%% Step 2: Estimation (MLE)
Y = Ym(2:end);
X = [Xm(2:end) YmT];
T = rows(Y);
k = cols(X);

% Estimation
theta0 = [0;0;0;0.1];
Data = [Y X];
index = 1:4;
index = index';

[thetamx, fmax, V, Vinv] = SA_Newton(@lnlik, @paramconst, theta0, Data, printi, index);

diag_V = diag(V);
stde = sqrt(diag_V);
t_val = thetamx./stde;
p_val = 2*(1 - cdf("t",abs(t_val),T-k));
