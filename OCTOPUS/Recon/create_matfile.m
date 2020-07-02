%%
% Dat2mat for ORC
% Author: Marina Manso Jimeno
% Last Updated: 02/28/2020

%% Raw data (Spiral)
addpath(genpath('../../../Other code/dat2mat'))
[Filename,Pathname] = uigetfile('*.dat','Pick the raw data file');
twix_obj = mapVBVDVE(fullfile(Pathname,Filename));
image_obj = twix_obj{2}.image; %{2}.image;
% TODO: add slices
sizeData = image_obj.sqzSize; %npoints, nch, nshots

dat = squeeze(image_obj(:,:,:));
dat = permute(dat, [1 3 2]);
save(strcat(Pathname,'rawdata_spiral.mat'), 'dat');

%% Field Map
addpath(genpath('../../../Other code/dat2mat'))
[Filename,Pathname] = uigetfile('*.dat','Pick the raw data file');
twix_obj = mapVBVDVE(fullfile(Pathname,Filename));
image_obj = twix_obj{2}.image; %{2}.image;
sizeData = image_obj.sqzSize; %lines, ch, columns, partitions

b0_map = squeeze(image_obj(:,:,:,:));
b0_map = permute(b0_map, [1 3 4 2]);

save(strcat(Pathname,'rawdata_b0map.mat'), 'b0_map');

%% Sampling Density Correction
addpath(genpath('../../../Other code/dcf_MATLAB'))
[Filename,Pathname] = uigetfile('*.mat','Pick the k-space trajectory file');
dcf = find_dcf(strcat(Pathname,Filename), true, strcat(Pathname,'dcf.mat'));

