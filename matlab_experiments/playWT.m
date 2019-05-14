function [out] = playWT(wt, f0, fs, t)

fs = 44100;

% epxecting instances of WT in 1st dimension
% and time vector in scond
K = size(wt,1); % number of instances
N = size(wt,2); % number of samples


n = t*fs;

% add a second period just to make stuff easier
wt = repmat(wt, [1,2]); 

p = 0;
k = 0;
% phase increment per sample
pI = f0/fs;
kI = K/(t*fs);

out = zeros(1,n);

for i = 1:n
    p = p + pI;
    k = k + kI;
    
    p = p - (p > 1);
    k = min(k, K-1.01); % lazy limit
    
    
    % integer and fractional parts for table lookup and interpolation
    x = p*N;
    xInt  = floor(x);
    xFrac = x-xInt;
    
    kInt = floor(k);
    kFrac = k-kInt;
    
    % 2d linear interpolation
    y1 = (1-xFrac) * wt(1+kInt,1+xInt) + (xFrac) * wt(1+kInt,2+xInt); 
    y2 = (1-xFrac) * wt(2+kInt,1+xInt) + (xFrac) * wt(2+kInt,2+xInt);
    y = (1-kFrac) * y1 + kFrac * y2;
    
    out(i) = y;
end




end
