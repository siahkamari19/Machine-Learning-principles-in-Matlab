function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  z   = X * theta;
  h_x = sigmoid(z);
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term;
  grad(1) = (1/m) * (X(:,1)'*(h_x-y));
  grad(2:end) = (1/m) * (X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end);
  grad = grad(:);
end

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_labels, n + 1);  
  X = [ones(m, 1) X];
  initial_theta = zeros(n+1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for c=1:num_labels
  all_theta(c,:) = ...
           fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                   initial_theta, options);
  end
end

function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  p = zeros(size(X, 1), 1);
  X = [ones(m, 1) X];
  prob_mat = X * all_theta';
  [prob, p] = max(prob_mat,[],2);
end

function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  a1 = [ones(m,1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 =  [ones(size(a2,1),1) a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  [prob, p] = max(a3,[],2); 
end