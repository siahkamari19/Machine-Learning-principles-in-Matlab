function [U, S] = pca(X)
  [m, n] = size(X);
  U = zeros(n);
  S = zeros(n);
  Sigma = (1/m)*(X'*X);
  [U, S, V] = svd(Sigma);
end

function Z = projectData(X, U, K)
  Z = zeros(size(X, 1), K);
  U_reduce = U(:,[1:K]);
  Z = X * U_reduce;
end

function X_rec = recoverData(Z, U, K)
  X_rec = zeros(size(Z, 1), size(U, 1));
  U_reduce = U(:,1:K);
  X_rec = Z * U_reduce';
end

function idx = findClosestCentroids(X, centroids)
  K = size(centroids, 1);
  idx = zeros(size(X,1), 1);
  for i = 1:size(X,1)
      temp = zeros(K,1);
      for j = 1:K
          temp(j)=sqrt(sum((X(i,:)-centroids(j,:)).^2));
      end
      [~,idx(i)] = min(temp);
  end
end

function centroids = computeCentroids(X, idx, K)
  [m n] = size(X);
  centroids = zeros(K, n);
  for i = 1:K
      idx_i = find(idx==i);
      centroids(i,:) = mean(X(idx_i,:));
  end
end

function centroids = kMeansInitCentroids(X, K)
  centroids = zeros(K, size(X, 2));
  randidx = randperm(size(X, 1));
  centroids = X(randidx(1:K), :);
end