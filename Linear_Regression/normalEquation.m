% ===================================
% name: gradientDescent.m
% author: Arthur Correnson
% mail: <arthur.correnson@gmail.com>
% ===================================

% === INIT ===

clc; close all; clear;

% === DATA ===

% load data file
load('data.txt');
% get input variables (features)
x = [ones(size(data, 1), 1), data(:, 1)];
% get output
y = data(:, size(data, 2));

% === LEARNING ===

% minimize the cost using normal equation
t = pinv(x' * x) * x' * y;
disp('');

% === RESULT ===

disp('Computed theta :'); 
disp(t);
disp('');

% compute the delta between predicted output and
% real output
delta = (x*t) - y;

% test the precision
p = 1e-5;
if (abs(delta) < p)
  printf('Delta is less than %d\n', p);
  printf('Prediction is reliable\n');
else
  printf('Delta is greater than %d\n', p);
  printf("Prediction isn't reliable\n", p);
endif;

% plot
plot(x(:, 2), y, '+r', 'markersize', 10);
hold on;

a = [ones(21, 1), (0:20)'];
plot(a(:, 2), a*t);