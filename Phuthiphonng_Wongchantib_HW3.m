clc;
clear;

% DH Transformation Function
function T = dh_transform(theta, d, a, alpha)
  T = [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
       sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
       0,           sin(alpha),             cos(alpha),            d;
       0,           0,                      0,                     1];
end

% Joint angles in radians
theta = deg2rad([0, 0, 0, 0, 0, 0, 0]);

% DH parameters
d     = [340, 0, 400, 0, 400, 0, 126];
a     = [0,   0,   0,   0,   0,  0,  0];
alpha = [pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0];

% Compute full transformation matrix
T = eye(4);
for i = 1:7
  T = T * dh_transform(theta(i), d(i), a(i), alpha(i));
end

% Display result
disp(T);

