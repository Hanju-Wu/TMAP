function funv = loss_smv(x, data)

% This function evaluates the function value of the smooth loss function

A = data.A;
b = data.b;
m = length(b);


Ax = A*x;
expba = exp(- b.*Ax);
funv = sum(log(1 + expba))/m;


end

