function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
