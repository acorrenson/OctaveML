clc; close all; clear;

load('dataSphere.txt');

% 100 training examples
global m = 100;
% 3 features (x, y, z)
global n = 3;

function B = mapFeatures(A)
  % xi² yi² zi²
  m = size(A, 1);
  quadTerms = A .^ 2;
  B = [ones(m, 1), A, quadTerms];
endfunction

function r = sig(z)
  r = 1 ./ (1 + exp(-z));
endfunction

% generate points :

X = zeros(m, n);
y = zeros(m, 1);

for i = 1:m
  a = rand * pi;
  b = rand * pi;
  r = rand * 10;
  l = 0;
  if r >= 5
    l = 1;
  end
  cx = cos(a) * cos(b) * r;
  cy = cos(b) * sin(a) * r;
  cz = sin(b) * r;
  X(i, 1:n) = [cx, cy, cz];
  y(i) = l;
endfor

csvwrite('dataSphere.txt', [X, y]);

X2 = mapFeatures(X);

iter = 200;
a = 0.1;
t = zeros(size(X2, 2), 1);

for i = 1:iter
  t = t - (a/m) .* X2' * (sig(X2*t) - y);
endfor


class1 = X(find(y == 1), :);
u = class1(:, 1)';
v = class1(:, 2)';
w = class1(:, 3)';
plot3(u, v, w, '+r');
hold on;

class2 = X(find(y == 0), :);
u = class2(:, 1)';
v = class2(:, 2)';
w = class2(:, 3)';
plot3(u, v, w, 'ob');
legend('out of the sphere', 'in the sphere');
hold on;

l = 20;

u = linspace(-5, 5, l);
v = linspace(-5, 5, l);
w = linspace(-5, 5, l);

score = zeros(l .^ 3, 4);
index = 0;

for i = 1:l
  for j = 1:l
    for k = 1:l
      index = index + 1;
      a = t(1);
      b = t(2);
      c = t(3);
      d = t(4);
      e = t(5);
      f = t(6);
      g = t(7);
      px = u(i);
      py = v(j);
      pz = w(k);
      r = a + b*px + c*py + d*pz + e*px.^2 + f*py.^2 + g*pz.^2;
      score(index, 4) = r;
      score(index, 1) = u(i);
      score(index, 2) = v(j);
      score(index, 3) = w(k); 
    end
  end
end

finalScore = find(ceil(score(:,4)) == 0);

finalX = score(finalScore, 1);
finalY = score(finalScore, 2);
finalZ = score(finalScore, 3);

plot3(finalX, finalY, finalZ, '.k', 'MarkerSize', 2);

%axis([-5 5 -5 5 -5 5]);

