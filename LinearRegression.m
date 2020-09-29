function plotData(x, y)
  figure;
  plot(x, y, 'rx', 'MarkerSize', 10);
  ylabel('Profit in $10,000s');
  xlabel('Population of City in 10,000s');
end
  
function J = computeCost(X, y, theta)
  m = length(y);
  J = 0;
  J = (1/(2*m))*sum(((X*theta)-y).^2);
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
   error = (X * theta) - y;
   theta = theta - ((alpha/m) * X'*error);
   J_history(iter) = computeCost(X, y, theta);
  end
end

function J = computeCostMulti(X, y, theta)
  m = length(y);
  J = 0;
  J = (1/(2*m))*(sum(((X*theta)-y).^2));
end

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
   error = (X * theta) - y;
   theta = theta - ((alpha/m) * X'*error);
   J_history(iter) = computeCostMulti(X, y, theta);
  end
end

function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  mu = mean(X);
  sigma = std(X);
  X_norm = (X - mu)./sigma;
end

function [theta] = normalEqn(X, y)
  theta = zeros(size(X, 2), 1);
  theta = pinv(X'*X)*X'*y;
end