% ===================================
% name: gradientDescent.m
% author: Arthur Correnson
% mail: <arthur.correnson@gmail.com>
% license: MIT
% ===================================

% === INIT ===

clc; close all; clear;

% === GENERATE DATA ===

% 100 training examples
global m = 100;
% 3 features (x, y, z)
global n = 3;

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

% save generated data
csvwrite('dataSphere.txt', [X, y]);

% === FUNCTIONS ===

% map features
function B = mapFeatures(A)
  % xi² yi² zi²
  m = size(A, 1);
  quadTerms = A .^ 2;
  B = [ones(m, 1), A, quadTerms];
endfunction

% sigmoid activation function
function r = sig(z)
  r = 1 ./ (1 + exp(-z));
endfunction


% === LEARNING ===

% map features
mapX = mapFeatures(X);

% number of iterations
iter = 200;
% learning rate
a = 0.1;
% init theta
t = zeros(size(mapX, 2), 1);

disp('running gradient descent...');
for i = 1:iter
  t = t - (a/m) .* mapX' * (sig(mapX*t) - y);
endfor
disp('...done');

% === RESULT ===

% display elements of class1
class1 = X(find(y == 1), :);
u = class1(:, 1)';
v = class1(:, 2)';
w = class1(:, 3)';
plot3(u, v, w, '+r');
hold on;

% display elements of class2
class2 = X(find(y == 0), :);
u = class2(:, 1)';
v = class2(:, 2)';
w = class2(:, 3)';
plot3(u, v, w, 'ob');
hold on;

legend('out of the sphere', 'in the sphere');

% Plot decision boundary

% number of points / axis
l = 20;
u = v = w = linspace(-5, 5, l);

% result of X * t for each points
% -- Xi = point (xi, yi, zi)
% rows are like : [xi yi zi ri] 
% -- ri = Xi * t
score = zeros(l .^ 3, 4);

% compute every (Xi*t)
index = 0;
for i = 1:l
  for j = 1:l
    for k = 1:l
      index = index + 1;
      coords = [u(i), v(j), w(k)];
      r = mapFeatures(coords) * t;
      score(index, 1:4) = [coords, r];
    end
  end
end

% keep only the points Xi that satisfy
% the equation Xi * t = 0 or : 
% ax + by + cz + ex² + fy² + gz² + h = 0
finalScorePos = find(ceil(score(:, 4)) == 0);

finalX = score(finalScorePos, 1);
finalY = score(finalScorePos, 2);
finalZ = score(finalScorePos, 3);

plot3(finalX, finalY, finalZ, '.k', 'MarkerSize', 2);


