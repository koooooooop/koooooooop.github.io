% MATLAB example code for MCM/ICM
function [output] = example_function(input)
    % This is a simple example function
    output = input.^2 + 2*input + 1;
    
    % Plot the results
    x = -10:0.1:10;
    y = x.^2 + 2*x + 1;
    plot(x, y);
    title('Example Function: f(x) = x^2 + 2x + 1');
    xlabel('x');
    ylabel('f(x)');
end

% Main script example
x = -5:5;
y = example_function(x);
disp('Results:');
disp([x; y]'); 