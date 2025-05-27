function [objective, xs] = GD(A, initial_x, b, max_iters, gamma, dim)

% variables definition
xs = zeros([max_iters,dim]); % solutions matrix
objective = zeros(max_iters,1); % objective values vector

% decision variable inizialization
xs(1,:) = initial_x;

for i = 1:max_iters

    % Compute objective and gradient

    % objective(i) = norm(A*xs(i,:)' - b).^2;
    % grad = 2*A'*(A*xs(i,:)' - b);

    [objective(i), grad] = Objective_and_Gradient_computation(A, xs(i,:), b);
 
    % Solution update

    xs(i+1,:) = xs(i,:) - gamma.*grad';

    % Display the results at each iteration

    fprintf('Iteration number: %3.0f\n', i);
    fprintf('   x = %3.7f %11.7f %11.7f %11.7f\n', xs(i,:));
    fprintf('   Objective = %3.5f\n', objective(i));

end

  



