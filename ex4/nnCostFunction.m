function [J,grad] = nnCostFunction(nn_params, input_layer_size,...
    hidden_layer_size, num_labels, X, Y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
% Setup some useful variables
m = size(X, 1);

% ==========================Forward Propagation============================
% Input Layer 
a_1 = [ones(m,1) X];

% 2nd Layer (Hidden Layer)
z_2 = a_1*Theta1';
a_2 = [ones(m,1) sigmoid(z_2)];

% %Output Layer
z_3 = a_2*Theta2';
h_theta = sigmoid(z_3);

%========================= Cost Function ==================================
% Recasts the label vector y as a 'Hot one out' boolean matrix where each 
% line corresponds to one example

% A little obfuscated but more vectorized, thus more eficient - what this
% does is index h_theta with the label Y, thus taking care of the internal
% sum, this can be done bc Y is boolean

J = (-sum(log(h_theta(Y))) - sum(log(1 - h_theta(~Y))))/m;

% Regularization term, just a plain double sum on theta.^2, we start from
% second column bc first column is the bias, which we don't sum over - this
% has been separated from the operation above just for clarity
J = J + 0.5*lambda/m*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


% ============================ Backpropagation ============================
% Calculating error delta terms
delta_3 = h_theta - Y;
delta_2 = delta_3*Theta2(:,2:end).*sigmoidGradient(z_2);

%Capital D's in this implementation are already summed over all training
%examples
D_1 = delta_2'*a_1;
D_2 = delta_3'*a_2;

% Normalizes capital D by # of training examples and adds regularization
% term
Theta1_grad = D_1/m + lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = D_2/m + lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end