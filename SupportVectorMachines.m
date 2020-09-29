function sim = gaussianKernel(x1, x2, sigma)
  x1 = x1(:); x2 = x2(:);
  sim = 0;
  sim = exp(-1*sum(abs(x1-x2).^2)/(2*sigma^2));
end

function [C, sigma] = dataset3Params(X, y, Xval, yval)
  C = 1;
  sigma = 0.3;
  C_list     = [0.01 0.03 0.1 0.3 1 3 10 30]';
  sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30]';
  prediction_error = zeros(length(C_list), length(sigma_list));
  result = zeros(length(C_list)+length(sigma_list),3);
  row = 1;
  for i = 1:length(C_list)
      for j = 1: length(sigma_list)
          C_test = C_list(i);
          sigma_test = sigma_list(j);
          model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
          predictions = svmPredict(model, Xval);
          prediction_error(i,j) = mean(double(predictions ~= yval));
          result(row,:) = [prediction_error(i,j), C_test, sigma_test];
          row = row + 1;
      end
  end
  sorted_result = sortrows(result, 1);
  C = sorted_result(1,2);
  sigma = sorted_result(1,3);
end

function word_indices = processEmail(email_contents)
  vocabList = getVocabList();
  word_indices = [];
  email_contents = lower(email_contents);
  email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
  email_contents = regexprep(email_contents, '[0-9]+', 'number');
  email_contents = regexprep(email_contents, ...
      '(http|https)://[^\s]*', 'httpaddr');
  email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
  email_contents = regexprep(email_contents, '[$]+', 'dollar');
  fprintf('\n==== Processed Email ====\n\n');
  l = 0;
  while ~isempty(email_contents)
    [str, email_contents] = ...
        strtok(email_contents, ...
        [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
    str = regexprep(str, '[^a-zA-Z0-9]', '');
    try str = porterStemmer(strtrim(str));
    catch str = ''; continue;
    end;
    if length(str) < 1
        continue;
    end
    index = find(strcmp(str,vocabList),1);
    word_indices = [word_indices; index];
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;
  end
  fprintf('\n\n=========================\n');
end

function x = emailFeatures(word_indices)
  n = 1899;
  x = zeros(n, 1);
  for i = 1:length(word_indices)
      x(word_indices(i)) = 1;
  end
end