% ===================================
% name: gradientDescent.m
% author: Arthur Correnson
% mail: <arthur.correnson@gmail.com>
% ===================================

% === INIT ===

close all; clc; clear;

% === DATA ===

% load data file
load('data2.txt');
% get input variables (features)
X = [ones(size(data2, 1), 1), data2(:, 1:2)];
% get output
y = data2(:, 3);

% === FUNCTIONS ===

% convert y output for one-vs-all classification
function y = labelsForClass (d, c)
  for i = 1:length(d)
    if (d(i) != c)
      d(i) = 0;
    else
      d(i) = 1;
    endif
  endfor
  y = d;
endfunction 

% sigmoid function
function r = sig(z)
  r = 1 ./ (1 + exp(-z));
endfunction

% predict the class for any input
% -- t = t1 -> "probability to belong to class 1" | p(y = 0)
% -- t = t2 -> "probability to belong to class 2" | p(y = 1)
% -- t = t3 -> "probability to belong to class 3" | p(y = 2)
% t can be a matrix of vectors:
% Predict(input, [t1, t2, t3])
% ans = 
%   p(y = 0)
%   p(y = 1)
%   p(y = 2)
function p = Predict(X, t)
  p = sig(X*t)';
endfunction

% === LEARNING ===

% init theta(s)
t1 = zeros(3, 1);
t2 = zeros(3, 1);
t3 = zeros(3, 1);

% learning rate
a = 1;
% size of the training set
m = size(data2, 1);

y1 = labelsForClass(y, 0);
y2 = labelsForClass(y, 1);
y3 = labelsForClass(y, 2);

% number of iterations
iter = 1000;

disp('running gradientDescent...');
for i=1:iter
  % gradient descent for class 1
  t1 = t1 - (a/m) .* (X' * (sig(X*t1) - y1));
  % gradient descent for class 2
  t2 = t2 - (a/m) .* (X' * (sig(X*t2) - y2));
  % gradient descent for class 3
  t3 = t3 - (a/m) .* (X' * (sig(X*t3) - y3));
endfor
disp('...done');

% === RESULT ===

T = [t1, t2, t3];

disp('Computed Theta for class 1 :');
disp(t1); disp('');
disp('Computed Theta for class 2 :');
disp(t2); disp('');
disp('Computed Theta for class 3 :');
disp(t3); disp('');

% display the training set
plot(X(1:3, 2), X(1:3, 3), "xr", "Markersize", 10);
hold on;
plot(X(4:6, 2), X(4:6, 3), "ob", "Markersize", 10);
hold on;
plot(X(7:9, 2), X(7:9, 3), "+g", "Markersize", 10);
hold on;

% plot decision boundaries
x = [-100 100];

a = - t1(2) / t1(3);
b = - t1(1) / t1(3);
plot(x, a * x + b, 'r');
hold on;

a = - t2(2) / t2(3);
b = - t2(1) / t2(3);
plot(x, a * x + b, 'b');
hold on;

a = - t3(2) / t3(3);
b = - t3(1) / t3(3);
plot(x, a * x + b, 'g');
hold on;

axis([0, 12, 0, 20]);