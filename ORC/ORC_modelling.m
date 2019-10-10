%%
%
%     1. Demonstrates Off-Resonance Modeling and Correction using CPR for numerical
%     phantoms
% 
%     
% 
%     Returns
%     -------
%       orc : struct
%           contains different data points in an orc cpr implementation
%       
% 
% Copyright of the Board of Trustees of Columbia University in the City of New York


addpath(genpath('.'));
%% Experimental conditions
M = zeros(16);
M(8,8) =1;
kt = 0:1e-3:3e-3; %s
df =250.*ones(size(M)); %Hz
kt_array = repmat(kt,[size(M,1) size(M,2)/length(kt)]).';


%% FWD model
M_fwd = add_or(M,kt_array,df);
figure(101); imagesc(cat(2, abs(M), abs(M_fwd))); drawnow;

%% Inverse model - ORC

M_hat = orc(M_fwd,kt_array,df);
figure(101); imagesc(cat(2, abs(M), abs(M_fwd), abs(M_hat))); drawnow;
title('Original object                      OR corrupted                               OR corrected');

