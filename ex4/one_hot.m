function out = one_hot(y)
% Recasts the vector of integers y as boolean matrix rec with the property
% that on the i_th line of rec, all elements are zero except the j_th 
% element where j = y(i)
%
% Used for formatting the labels of a dataset for training a multiclass
% classification neural network

% Number of examples
m = size(y,1);

% Number of classes, note that this is only correct if there is a '0' class
% that does not map to a 0 and all the classes map to all the integers up
% to the number of classes without skipping any and all classes are
% represented in the example dataset.
K = max(y);

out = (kron(1:K,ones(m,1)) == kron(y,ones(1,K)));
end