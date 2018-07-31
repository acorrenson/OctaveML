
load('data2.txt');

X = [ones(size(data2, 1), 1), data2(:, 1:2)];
y = data2(:, 3);

function y = dataSet (d, c)
  for i = 1:length(d)
    if (d(i) != c)
      d(i) = 0;
    else
      d(i) = 1;
    endif
  endfor
  y = d;
endfunction 


function r = sig(z)
  r = 1 ./ (1 + exp(-z));
endfunction

function p = Predict(X, t)
  p = sig(X*t);
endfunction

iter = 1000;

t = zeros(3, 1);
t2 = zeros(3, 1);
t3 = zeros(3, 1);

a = 1;
m = size(data, 1);

y1 = dataSet(y, 0);
y2 = dataSet(y, 1);
y3 = dataSet(y, 2);

for i=1:iter
  % gradient descent for class 1
  t = t - (a/m) .* (X' * (sig(X*t) - y1));
  % gradient descent for class 2
  t2 = t2 - (a/m) .* (X' * (sig(X*t2) - y2));
  % gradient descent for class 3
  t3 = t3 - (a/m) .* (X' * (sig(X*t3) - y3));
endfor


disp(Predict(X, t2));

plot(X(:, 2), X(:, 3), "xr", "Markersize", 10);
hold on;

x = [1 10];

a = - t(2) / t(3);
b = - t(1) / t(3);
plot(x, a * x + b);
hold on;

a = - t2(2) / t2(3);
b = - t2(1) / t2(3);
plot(x, a * x + b);
hold on;

a = - t3(2) / t3(3);
b = - t3(1) / t3(3);
plot(x, a * x + b);
hold on;

axis([0, 12, 0, 20]);