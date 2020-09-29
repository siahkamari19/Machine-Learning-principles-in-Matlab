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
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  z = X * theta;
  h_x = sigmoid(z);
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x)));
  grad = (1/m)* (X'*(h_x-y));
end

function p = predict(theta, X)
  m = size(X, 1);
  p = zeros(m, 1);
  h_x = sigmoid(X*theta);
  p=(h_x>=0.5);
end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  z = X * theta;
  h_x = sigmoid(z); 
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term;
  grad(1) = (1/m)* (X(:,1)'*(h_x-y));
  grad(2:end) = (1/m)* (X(:,2:end)'*(h_x-y))+(lambda/m)*theta(2:end);
end