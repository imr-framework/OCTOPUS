function M_hat = orc(M,kt,df)

%% Create the phase matrix - 2*pi*df*t for every df and every t
kspace = fftshift(fft2(M));
M_hat = zeros(size(M));
for x = 1:size(M,1)
    for y = 1:size(M,2)
   
        phi = 2.*pi.*df(x,y).*kt;
        kspace_orc = kspace.*exp(1i*phi);
        M_corr = ifft2(kspace_orc);
        M_hat(x,y) = M_corr(x,y);
        
    end
end
