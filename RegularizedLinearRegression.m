function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  h_x = X * theta; % 12x1
  J = (1/(2*m))*sum((h_x - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2); % scalar
  grad(1) = (1/m)*(X(:,1)'*(h_x-y)); % scalar == 1x1
  grad(2:end) = (1/m)*(X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end); % n x 1
  grad = grad(:);
end

function [error_train, error_val] = ...
  learningCurve(X, y, Xval, yval, lambda)
  m = size(X, 1);
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);
  for i = 1:m
   Xtrain = X(1:i,:);
   ytrain = y(1:i);
   theta = trainLinearReg(Xtrain, ytrain, lambda);
   error_train(i) = linearRegCostFunction(Xtrain, ytrain, theta, 0); %for lambda = 0;
   error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0); %for lambda = 0;
  end
end

function [X_poly] = polyFeatures(X, p)
  X_poly = zeros(numel(X), p); % m x p
  X_poly(:,1:p) = X(:,1).^(1:p); % w/o for loop
end

function [lambda_vec, error_train, error_val] = ...
  validationCurve(X, y, Xval, yval)
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  m = size(X, 1);
  for j = 1:length(lambda_vec)
   lambda = lambda_vec(j);
   theta = trainLinearReg(X, y, lambda);
   error_train(j) = linearRegCostFunction(X, y, theta, 0); % lambda = 0;
   error_val(j)   = linearRegCostFunction(Xval, yval, theta, 0); % lambda = 0
  end
endfunction [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  h_x = X * theta; % 12x1
  J = (1/(2*m))*sum((h_x - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2); % scalar
  grad(1) = (1/m)*(X(:,1)'*(h_x-y)); % scalar == 1x1
  grad(2:end) = (1/m)*(X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end); % n x 1
  grad = grad(:);
end

function [error_train, error_val] = ...
  learningCurve(X, y, Xval, yval, lambda)
  m = size(X, 1);
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);
  for i = 1:m
   Xtrain = X(1:i,:);
   ytrain = y(1:i);
   theta = trainLinearReg(Xtrain, ytrain, lambda);
   error_train(i) = linearRegCostFunction(Xtrain, ytrain, theta, 0); %for lambda = 0;
   error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0); %for lambda = 0;
  end
end

function [X_poly] = polyFeatures(X, p)
  X_poly = zeros(numel(X), p); % m x p
  X_poly(:,1:p) = X(:,1).^(1:p); % w/o for loop
end

function [lambda_vec, error_train, error_val] = ...
  validationCurve(X, y, Xval, yval)
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  m = size(X, 1);
  for j = 1:length(lambda_vec)
   lambda = lambda_vec(j);
   theta = trainLinearReg(X, y, lambda);
   error_train(j) = linearRegCostFunction(X, y, theta, 0); % lambda = 0;
   error_val(j)   = linearRegCostFunction(Xval, yval, theta, 0); % lambda = 0
  end
end