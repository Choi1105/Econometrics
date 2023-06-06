function [lnL] = lnlik(theta,Data)

Y = Data(:,1);
X = Data(:,2:end);
T = rows(Y);


beta = theta(1:3);
sig2 = theta(4);

lnLm = zeros(T,1);
for t = 1:T

    lnLm(t) = log(mvnpdf(Y(t),X(t,:)*beta,sig2));
    
end

lnL = sum(lnLm);
end