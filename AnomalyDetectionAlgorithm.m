function [mu sigma2] = estimateGaussian(X)
  [m, n] = size(X);
  mu = zeros(n, 1);
  sigma2 = zeros(n, 1);
  mu = ((1/m)*sum(X))';
  sigma2 = ((1/m)*sum((X-mu').^2))';
end

function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;
  stepsize = (max(pval) - min(pval)) / 1000;
  for epsilon = min(pval):stepsize:max(pval)
      cvPredictions = (pval < epsilon);     % m x 1 
      tp = sum((cvPredictions == 1) & (yval == 1)); % m x 1
      fp = sum((cvPredictions == 1) & (yval == 0)); % m x 1
      fn = sum((cvPredictions == 0) & (yval == 1)); % m x 1
      prec = tp/(tp+fp); 
      rec = tp/(tp+fn);
      F1 = 2*prec*rec / (prec + rec);
      if F1 > bestF1
         bestF1 = F1;
         bestEpsilon = epsilon;
      end
  end
end

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  J = 0;
  X_grad = zeros(size(X));            % Nm x n
  Theta_grad = zeros(size(Theta));    % Nu x n
  %Without Regularization
  Error = (X*Theta') - Y;
  J = (1/2)*sum(sum(Error.^2.*R));
  X_grad = (Error.*R)*Theta;   % Nm x n
  Theta_grad = (Error.*R)'*X;  % Nu x n
  
  %With Regularization
  Reg_term_theta = (lambda/2)*sum(sum(Theta.^2));
  Reg_term_x = (lambda/2)*sum(sum(X.^2));
  J = J + Reg_term_theta + Reg_term_x;
  X_grad = X_grad + lambda*X;             % Nm x n
  Theta_grad = Theta_grad + lambda*Theta; % Nu x n
  grad = [X_grad(:); Theta_grad(:)];
end