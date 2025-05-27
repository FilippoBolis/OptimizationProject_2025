function [obj, gradient] = Objective_and_Gradient_computation(A, x, b)

obj = norm(A*x' - b).^2; % Objective evaluation
gradient = 2*A'*(A*x' - b); % Gradient computation