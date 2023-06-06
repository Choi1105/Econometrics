%% Simultaneous Structural Equation Model
%  Model 
%  y_1(t) = alpha1*y_2(t) + alpha2*x_1(t) + e_1(t)
%  y_2(t) = beta1*y_t(t) + beta2*x_2(t) + e_2(t)
%  whete (e_1(t) e_2(t))' = e(t) , where e(t) ~ N(0 , Sigma), k by 1

clear
clc

%% STEP 1: Data Generating Process
% Data inform
T = 1000;
k = 2;
printi = 0;

% True Param
alpha1 = 1;
alpha2 = 2;
beta1 = 2;
beta2 = 3;

% True Sig = V*Gam*V, (k by k)
% Where Gam = correlatin Mattix / V = Volatility matrixn (diagonal)
Gam = [1 , 0 ; 0 , 1];
V = 0.9*eye(k);
Sig = V*Gam*V;

% Exogenous Variables in each regression
x1m = rand(T,1)*5;
x2m = rand(T,1)*5;
xm = [x1m , x2m];

% Pre allocation for ym
ym  = zeros(T,k);

% Structural param in matrix
B = [1 -alpha1 ; -beta1 1];
C = [alpha2 0 ; 0 beta2];

% Reduced form parameters in matrix
P = inv(B)*C;
Omega = inv(B)*Sig*inv(B)';

for i = 1:T

    % At each time t, exo variable
    xt = xm(i,:)';

    % At each time t, dependent variable, k by 1
    yt = P*xt + chol(Omega)'*randn(k,1);
    ym(i,:) = yt';
end
%% STEP 2: Estimation by OLS
% 1st Equation
Y1 = ym(:,1);
X1 = [ym(:,2) x1m];
[a_hat_OLS, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,X1,printi);

% 2nd Equation
Y2 = ym(:,2);
X2 = [ym(:,1) x2m];
[b_hat_OLS, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,X2,printi);
ab_hat_OLS = [a_hat_OLS; b_hat_OLS];

%% STEP 3: Estimation by 2SLS
% Instrumental Variables
Z = [x1m x2m];

% Stage 1: Estimate Y1hat, Y2hat
[~, ~, ~, ~, Y2hat, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,Z,printi);
[~, ~, ~, ~, Y1hat, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,Z,printi);

% Stage 2: Y1 -> Y1hat, Y2 -> Y2hat
% 1st Equation
Y1 = ym(:,1);
X1 = [Y2hat x1m];
[a_hat_2SLS, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,X1,printi);

% 2nd Equation
Y2 = ym(:,2);
X2 = [Y1hat x2m];
[b_hat_2SLS, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,X2,printi);
ab_hat_2SLS = [a_hat_2SLS; b_hat_2SLS];

%% STEP 4: Estimation by ILS
% Reduced form equation
% 1st Equation
Y1 = ym(:,1);
X1 = [x1m x2m];
[pi1hat, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y1,X1,printi);
pi11hat = pi1hat(1);
pi12hat = pi1hat(2);

% 2nd Equation
Y2 = ym(:,2);
X2 = [x1m x2m];
[pi2hat, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = OLSout(Y2,X2,printi);
pi21hat = pi2hat(1);
pi22hat = pi2hat(2);

% Compute Structural parameters
a1hat_ILS = pi12hat/pi22hat;
b1hat_ILS = pi21hat/pi11hat;
a2hat_ILS = pi11hat*(1 - a1hat_ILS*b1hat_ILS);
b2hat_ILS = pi22hat*(1 - a1hat_ILS*b1hat_ILS);

ab_hat_ILS = [a1hat_ILS;a2hat_ILS;b1hat_ILS;b2hat_ILS];

%% STEP 5: Results
disp("==================================================================")
disp(['OLS Estimator is   ', num2str(ab_hat_OLS')]);
disp(['2SLS Estimator is  ', num2str(ab_hat_2SLS')]);
disp(['ILS Estimator is   ', num2str(ab_hat_ILS')]);
disp("==================================================================")




