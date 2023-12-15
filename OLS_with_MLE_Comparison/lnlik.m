function [lnL] = lnlik(theta,Data)

Y = Data(:,1);
X = Data(:,2:end);
T = rows(Y);


beta = theta(1:2);
rho = theta(3);
sig2 = theta(4);

lnLm = zeros(T,1);

for t = 1:T
    
    lnLm(t) = log(mvnpdf(Y(t),(1-rho)*beta(1) + (1-rho)*X(t,1)*beta(2) + rho*X(t,2),sig2));
    
end

lnL = sum(lnLm);
end