function plotData(X, y)
  pos = find(y==1);
  neg = find(y==0);
  figure;
  plot(X(pos,1),X(pos,2),'g+');
  hold on;  
  plot(X(neg,1),X(neg,2),'ro');
  hold off;
end

function g = sigmoid(z)
  g = zeros(size(z));
  g = 1./(1+exp(-z));
end

function [J, grad] = costFunction(theta, X, y)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  z = X * theta;      % m x 1
  h_x = sigmoid(z);   % m x 1 
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))); % scalar
  grad = (1/m)* (X'*(h_x-y));     % (n+1) x 1
end

function p = predict(theta, X)
  m = size(X, 1); % Number of training examples
  p = zeros(m, 1);
  h_x = sigmoid(X*theta);
  p=(h_x>=0.5);
end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  z = X * theta;      % m x 1
  h_x = sigmoid(z);  % m x 1 
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; % scalar
  grad(1) = (1/m)* (X(:,1)'*(h_x-y));                                  % 1 x 1
  grad(2:end) = (1/m)* (X(:,2:end)'*(h_x-y))+(lambda/m)*theta(2:end);  % n x 1
end