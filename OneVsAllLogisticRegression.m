function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  z   = X * theta;   % m x 1
  h_x = sigmoid(z);  % m x 1 
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; % scalar
  grad(1) = (1/m) * (X(:,1)'*(h_x-y));                                    % 1 x 1
  grad(2:end) = (1/m) * (X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end);  % n x 1
  grad = grad(:);
end

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);        % No. of Training Samples == No. of Images : (Here, 5000) 
  n = size(X, 2);        % No. of features == No. of pixels in each Image : (Here, 400)
  all_theta = zeros(num_labels, n + 1);  
  X = [ones(m, 1) X];   %DIMENSIONS: X = m x (input_layer_size+1) = m x (no_of_features+1)
  initial_theta = zeros(n+1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for c=1:num_labels
  all_theta(c,:) = ...
           fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                   initial_theta, options);
  end
end

function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);     % No. of Input Examples to Predict (Each row = 1 Example)
  num_labels = size(all_theta, 1); %No. of Ouput Classifier
  p = zeros(size(X, 1), 1);    % No_of_Input_Examples x 1 == m x 1
  X = [ones(m, 1) X];
  prob_mat = X * all_theta';     % 5000 x 10 == no_of_input_image x num_labels
  [prob, p] = max(prob_mat,[],2); % m  x 1 
end

function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);  % m x 1
  a1 = [ones(m,1) X]; % 5000 x 401 == no_of_input_images x no_of_features % Adding 1 in X 
  z2 = a1 * Theta1';  % 5000 x 25
  a2 = sigmoid(z2);   % 5000 x 25
  a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26
  z3 = a2 * Theta2';  % 5000 x 10
  a3 = sigmoid(z3);  % 5000 x 10
  [prob, p] = max(a3,[],2); 
end