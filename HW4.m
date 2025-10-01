clc;
clear;

% DH Transformation Function
function R = rotage_axis_euler_ZYX(alpha, beta, gamma)
  R = [cos(gamma)*cos(beta), cos(gamma)*sin(beta)*sin(alpha)-cos(alpha)*sin(gamma),  sin(gamma)*sin(alpha)+cos(gamma)*cos(alpha)*sin(beta);
       cos(beta)*sin(gamma),  cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha), cos(alpha)*sin(gamma)*sin(beta)-cos(gamma)*sin(alpha);
       -sin(beta),             cos(beta)*sin(alpha),                                     cos(alpha)*cos(beta)];
end

alpha = pi/2;
beta = -pi/4;
gamma = pi/6;

R = rotage_axis_euler_ZYX(alpha, beta, gamma);

% Display result
disp('Rotation matrix R (ZYX Euler angles):');
disp(R);

