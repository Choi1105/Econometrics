%% GLS under Heteroskedasticity
clear;
clcl;

%% STEP .1 : DGP
T = 1000;
tau = 400;
beta1 = 1;
beta2 = 2;
sig12 = 1;
sig22 = 5;

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
[bhat_OLS, sig2hat_OLS, stde, ~, ~, ~, varbhat_OLS, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y,X,printi);