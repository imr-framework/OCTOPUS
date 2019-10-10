%% Understanding pixel shift
M1 = zeros(16);
[len_x, len_y] = size(M1);
x0 = 8;
y0 = 8;
M1(x0,y0) = 1;
kspace1 = fftshift(fft2(M1));



M2 = zeros(16);
x1 = 12;
y1 = y0;
M2(x1,y1) = 1;
kspace2 = fftshift(fft2(M2));

%% Calculating phase changes because of this pixel shift

dx = x1 - x0;
num_cycles_x = ceil(abs(len_x/dx));
phase_dx = sign(dx)*2*pi/num_cycles_x;


phases_x = 0:phase_dx:(num_cycles_x-1)*phase_dx; %Should be a column vector for x and a row vector for y
phase_matx = repmat(phases_x.',[ len_x/num_cycles_x  len_y ]);

%% Some viz 
kspace2_hat = kspace1.*exp(-1i.*phase_matx);
M2_hat = ifft2(kspace2_hat);
figure; imagesc(cat(2,abs(M1),abs(M2),abs(M2_hat)));
% Phase 
figure; imagesc(cat(2, rad2deg(mod(angle(kspace1) - angle(kspace2),2*pi)), zeros(len_x,1),rad2deg(mod(phase_matx,2*pi)))); 
