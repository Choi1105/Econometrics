%% Seemingly Unrelated Regression Model

%  Model 
%  y_1(t) = x_1(t)' * beta1 + e_1(t)
%  y_2(t) = x_2(t)' * beta2 + e_2(t)
%  whete x_1(t) = x_2(t) = ki by 1
%        (e_1(t) e_2(t))' = e(t) , where e(t) ~ N(0 , Omega), p by 1

clear
clc

%% STEP 1: Data Generating Process
% Data Inform
% Sample size (T) , Number of regression (P)
% Number of coefficient in each regression i(ki), total coefficient(k)
T = 1000;

p = 2;
k1 = 2;
k2 = 2;
k = k1 + k2;

% True beta 
b1 = [1;2];
b2 = [3;4];
beta = [b1;b2];

% True Omega = V*Gam*V (p by p)
% where Gam = Correlation Matrix / V = Volatility Matrix (diagonal)
Gam = [1, 0.8 ; 0.8, 1];
V = 0.9*eye(p);
Omega = V*Gam*V;

% Exogenous variables in each regression
x1m = [ones(T,1) rand(T,1)*5];
x2m = [ones(T,1) rand(T,1)*5];

% Pre-allocation for ym
ym = zeros(T,p);

for t = 1:T
    % at each time t, exogeneous variable matrix, p by k
    xt = zeros(p,k);
    xt(1,1:k1) = x1m(t,:);
    xt(2,(k1+1):end) = x2m(t,:);

    % at each time t, dependent variables, p by 1
    yt = xt*beta + chol(Omega)'*randn(p,1);
    ym(t,:) = yt';

end

%% STEP 2: Estimation by OLS
% 1st Equation
Y1 = ym(:,1);
X1 = x1m;

printi = 0;
[b1hat, sig12hat, ~, ~, ~, e1hat, varb1hat, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,X1,printi);

% 2nd Equation
Y2 = ym(:,2);
X2 = x2m;

printi = 0;
[b2hat, sig22hat, ~, ~, ~, e2hat, varb2hat, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,X2,printi);

%% STEP 3: Estimation by SUR
% Data Information
T = rows(Y1);
k1 = cols(X1);
k2 = cols(X2);
k = k1 + k2;

% Data 
Y = [Y1;Y2];
X = [X1, zeros(T,k2); zeros(T,k1), X2];

% Consturction Omega
Cov_hat = e1hat'*e2hat/(T-(k1+k2)/2);
Sigma = [sig12hat, Cov_hat; Cov_hat sig22hat];
Omega_hat = kron(Sigma, eye(T));

% SUR Estimator
bhat_SUR = inv(X'*inv(Omega_hat)*X)*X'*inv(Omega_hat)*Y;
varbhat_SUR = inv(X'*inv(Omega_hat)*X);

%% STEP 4: Results
disp('=====================================')
disp(['SUR estimator is  ', num2str(bhat_SUR')]);
disp(['SUR variance is  ', num2str(diag(varbhat_SUR)')]);
disp('=====================================')
disp(['OLS estimator is  ', num2str([b1hat',b2hat'])]);
disp(['OLS variance is  ', num2str([diag(varb1hat)', diag(varb2hat)'])]);
disp('=====================================')
