close all;
clc;
clear vars,
clear all;

cd 'C:\Users\Medolago\Desktop\Tutorati Optimization\Matlab codes\GD Matlab codes'

% ------ Data loading and preparation -------------------------------------

Dataset = readtable("TABELLA_NO.csv");
Data_sz = size(Dataset{:,1:4}); % dataset dimensions considering only the selected features and the dependent variable

A = [ones(Data_sz(1),1) Dataset{:,1:3}]; % first columns made up of one + three columns containing the regressors' observations
b = Dataset{:,4}; % dependent variable's observations
A_sz = size(A); % matrix A dimensions
dim = A_sz(2); % length of the solution vector x (number of columns in the matrix A), corresponding to x belonging to R^dim
% x contains the coefficients of the regression

% ------ Inizializations --------------------------------------------------

initial_x = zeros(1, dim); % x inizialization
max_iters = 10000; % max number of iterations

% ------ Learning rate ----------------------------------------------------

% % Naive Learning rate
% gamma = 0.000001; 

% % Learning rate assuming bounded gradient
% R = 8;
% B = 2*(R*norm(A'*A) + norm(A'*b));
% gamma = R/(B*max_iters.^(1/2));

% Learning rate assuming smoothness
L = 2*norm(A'*A);
gamma = 1/L;

% ------ Compute Gradient Descent -----------------------------------------

[objective, x] = GD(A, initial_x, b, max_iters, gamma, dim);

model_hp = fitlm(Dataset{:,1:3}, Dataset{:,4});

y_hat = A*x(end,:)';


%% With Feauture Standardization ------------------------------------------

standardized_regressors = normalize(Dataset{:,1:3});
A = [ones(Data_sz(1),1) standardized_regressors]; 
b = Dataset{:,4}; 

initial_x = zeros(1, dim);
max_iters = 500; 

L = 2*norm(A'*A);
gamma = 1/L;

% ------ Compute Gradient Descent -----------------------------------------

[objective, x] = GD(A, initial_x, b, max_iters, gamma, dim);

model_hp = fitlm(standardized_regressors, Dataset{:,4});

y_hat = A*x(end,:)';



