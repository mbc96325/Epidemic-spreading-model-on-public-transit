function [p_i,p_e] = simplified_model_dynamics(parameters_e,parameters_i,coefs_e,coefs_i,pi0,pe0,transit_rate)
% to use this function, people need to prepare
% the dynamics parameters for disease process
% the coeficients should be fitted use the data generated by the numerical
% model
% paremeters and coefs are factors of the regression model correspondingly.
% we also need the initial condition for the disease process (pi0,pe0).
% the transit rate is the probability for people to transit from infected
% to recovered
% the model should be scaled to fit different network capacity by scaling the input parameters and initial conditions.


p_e = pe0;
p_i = pi0;
rate1(1) = 1;
for i = 1:671
p_e(i+1) = p_e(i) + sum(coefs_e .* parameters_e(:,i));
p_i(i+1) = p_i(i) + sum(coefs_i .* parameters_i(:,i));
rate1(i+1) = rate1(i)*(1- transit_rate);
end
end
