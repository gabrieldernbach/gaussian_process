clear all; close all;


% testing a seedable RNG generating an equal distribution


n = 10000;

out = generateXORShift(n);

figure
histfit(out);


figure
plot(out(1:100));

figure
plot(mag2db(abs(fft(out))));




function [out] = generateXORShift(n)

    out = zeros(1,n);

    x = uint32(88172645463325252);
    
    for i = 1:n
    
        x = bitxor(x,bitshift(x, 13));
        x = bitxor(x,bitshift(x, -7));
        x = bitxor(x,bitshift(x, 17));
         
        out(i) = single(x) / 2.^64;
    end
end