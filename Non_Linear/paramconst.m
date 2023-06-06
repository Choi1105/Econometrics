function [valid] = paramconst(theta,Data)

validm = ones(10,1);

validm(1) = theta(4) > 0; 

valid = min(validm); 

end