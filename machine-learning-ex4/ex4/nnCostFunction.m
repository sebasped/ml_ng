function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X]; %agrego el bias al input layer
z2 = Theta1*a1';
a2 = [ones(1,m); sigmoid(z2)]; %agrego el bias al hidden layer
z3 = Theta2*a2;
a3 = sigmoid(z3); % output layer
h = a3;

%y = [repmat(eye(10)(:,10),1,500), repmat(eye(10)(:,1),1,500), repmat(eye(10)(:,2),1,500), repmat(eye(10)(:,3),1,500), repmat(eye(10)(:,4),1,500), repmat(eye(10)(:,5),1,500), repmat(eye(10)(:,6),1,500), repmat(eye(10)(:,7),1,500), repmat(eye(10)(:,8),1,500), repmat(eye(10)(:,9),1,500)];

yBinaria = zeros(num_labels,m);
for i=1:m
  yBinaria(:,i) = eye(num_labels)(:,y(i));
endfor

%y = yBinaria;
J = sum( -diag(yBinaria'*log(h)) - diag((ones(num_labels,m)-yBinaria)'*log(ones(num_labels,m)-h)) )/m;
%agrego regularización (la col 1, que es el bias, no se regulariza)
J = J + (lambda/(2*m))*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );


% back propagation para implementar el gradiente
d3 = a3 - yBinaria;
d2 = (Theta2'*d3) .* (a2.*(1-a2));

T2 = Theta2.*0;
T1 = Theta1.*0;

T2 = T2 + d3*a2';
T1 = T1 + d2(2:end,:)*a1;

Theta1_grad = T1/m;
Theta2_grad = T2/m;

% agrego regularización
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
