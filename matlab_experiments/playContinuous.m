function [out] = playContinuous(yn, sn, f0, fs, t, interval)
% generates new WT instances in real time

% interval: time in seconds between wt updates


fs = 44100;

% epxecting instances of WT in 1st dimension
% and time vector in scond
N = length(yn); % number of samples


n = t*fs;

% add a second period just to make stuff easier

wt1 = instanceWT(yn,sn);
wt2 = instanceWT(yn,sn);

wt1 = repmat(wt1, [1,2]);
wt2 = repmat(wt2, [1,2]);


p = 0;
k = 0;

% phase increment per sample

if(length(f0) > 1)
    f0Vec = 2.^interp1(linspace(0,1,length(f0)), log2(f0), linspace(0,1,n));
    pI = f0Vec / fs;
else
    pI = ones(1,n) .* (f0/fs);    
end

if(length(interval) > 1)
    intervalVec = 2.^(interp1(linspace(0,1,length(interval)), log2(interval), linspace(0,1,n)));
    kI = 1./(intervalVec * fs);
else
    kI =  ones(1,n) * 1/(interval*fs);    
end

out = zeros(1,n);

for i = 1:n
    p = p + pI(i);
    k = k + kI(i);
    
    p = p - (p > 1);
    
    if(k >= 1)
        wt1 = wt2;
        wt2 = instanceWT(yn,sn);
        wt2 = repmat(wt2, [1,2]);
        k = k - 1;
    end
    
    % integer and fractional parts for table lookup and interpolation
    x = p*(N-1);
    xInt  = floor(x);
    xFrac = x-xInt;
    
    kInt = floor(k);
    kFrac = k-kInt;
    
    % 2d linear interpolation
    y1 = (1-xFrac) * wt1(1+xInt) + (xFrac) * wt1(2+xInt); 
    y2 = (1-xFrac) * wt2(1+xInt) + (xFrac) * wt2(2+xInt); 
    y = (1-kFrac) * y1 + kFrac * y2;
    
    gain = db2mag(30);
    out(i) = gain*tanh( y/gain);
end




end
