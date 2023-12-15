clear
clc

% F-Test Method 3가지
% 1 : 가설을 일반화된 형태로 표현 후 가설검정
% 2 : 제약하의 회귀식과 비제약하의 회귀식을 돌린 다음 잔차항의 차이로 검정
% 3 : 상수항을 제외한 나머지 계수가 0인지 아닌지 판단 -> R^2사용

% Model
% Y = X1*B1 + X2*B2 + X3*B3 + X4*B4 + e  e ~ iidN(0,sig2)
% H0 : B2 = B3 = 0 , B4 - B5 = 0   
% -> P-value < 0.05 -> 기각 ( b2 = b3 =! 0, b4 - b5 =! 0 ) 
% 하나라도 어긋나면 기각됨.


% H1 : B2 =! B3 =! 0 , B4 - B5 =! 0

%F_MA =zeros(10000,3);
%P_MA =zeros(10000,3);

%% Step 1: DGP

T = 10000;
B1 = 6.1;
B2 = 0;
B3 = 0;
B4 = 6.6;
B5 = 6.6;
sig2 = 0.9;

X1m = ones(T,1);
X2m = 5*rand(T,1);
X3m = 5*rand(T,1);
X4m = 5*rand(T,1);
X5m = 5*rand(T,1);

em = sqrt(sig2)*randn(T,1);

Ym = X1m*B1 + X2m*B2 + X3m*B3 + X4m*B4 + X5m*B5 + em;

%% Step 2: Estimation (OLS)
% (Unrestricted Regression)
Y = Ym;
T = rows(Y);
X = [X1m X2m X3m X4m X5m];
k = 5;

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

%% Step 3: F-Test

% F-Test
R = [0 1 0 0 0 ; 0 0 1 0 0 ; 0 0 0 1 -1 ];
gam = [0 ; 0 ; 0];
k = rows(bhat);
L = rows(R);

Rbhat = R*bhat - gam;
Rvar = R*varbhat*R';
F_val = Rbhat'*invpd(Rvar)*Rbhat/L;
P_val_F = 1 - cdf('f',F_val,L,T-k);

% Wald Test, Residual Based Test
% H0 : B2 = B3 = 0 , B4 - B5 = 0 -> 이게 True라 가정.


Y_R = Ym;
X_R = [X1m*B1 (X4m+X5m)];

bhat_R = inv(X_R'*X_R)*X_R'*Y_R;
Yhat_R = X_R*bhat_R;
ehat_R = Y_R - Yhat_R;

ehat2 = ehat'*ehat;
ehat_R2 = ehat_R'*ehat_R;

Wald_nu = (ehat_R2 - ehat2)/L;
wald_de = ehat2/(T-k);
Wald = Wald_nu/wald_de;
P_val_Wald = 1 - cdf('f', F_val, L , T - k);

%% Step 4: Results
disp('==========================================');
disp('Joint Hypothesis');
disp('==========================================');
disp(['Test Type      ', '      F_value',  '            P_value']);
disp(['    ','F - Test       ','     ',num2str(F_val),'     ', num2str(P_val_F)]);
disp(['','Wald - Test    ','    ',num2str(Wald),'     ', num2str(P_val_Wald)]);
%disp(['   ','R2 - Test      ','    ',num2str(Wald_R2),'     ', num2str(P_val_Wald_R2)]);
