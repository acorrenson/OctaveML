% ===================================
% name: gradientDescent.m
% author: Arthur Correnson
% mail: <arthur.correnson@gmail.com>
% ===================================

% === INIT ===

clc; close all; clear;

% === DATA ===

% load data file
load ('data.txt');
% get input variables (features)
X = [ones(size(data, 1), 1), data(:, 1)];
% get output
y = data(:, size(data, 2));
% init theta
t = zeros(size(X, 2), 1);

% learning rate
a = 0.1;
% size of the training set
m = size(data, 1);
% precision
p = 1e-5;

% === LEARNING ===

% perform gradient descent
disp('Running gradientDescent...');
while true,
  E = X * t - y;
  if (abs(E) <= p)
    break; 
  endif;
  t = t - (a/m) .* (X' * E);
endwhile;
disp('...Done');

% === RESULT ===

disp('Computed Theta :');
disp(t);

plot(X(:, 2), y, 'xr', 'Markersize', 10);
hold on;
a = [ones(11, 1), (0:10)'];
plot(a(:, 2), a*t);
xlabel('input');
ylabel('prediction');