function smg = loss_smg(x, data)

% This function returns the gradient of the smooth loss function.


A = data.A;
b = data.b;
m = length(b);

Ax = A*x;
expba = exp(- b.*Ax);
smg = A'*(b./(1+expba) - b)/m;

end

