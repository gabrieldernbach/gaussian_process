clear all; 
close all;


acc  = 0.28;
damp = 0.8;
draw = acc;

n = ceil(10 / acc)

samples = [ drunkAcc(n,acc,damp,draw); ...
            drunkAcc(n,acc,damp,draw); ...
            drunkAcc(n,acc,damp,draw); ...
            drunkAcc(n,acc,damp,draw); ...
            drunkAcc(n,acc,damp,draw)];


        
plot(samples');
ylim([-5, +5])

histSamples = drunkAcc(100000, acc, damp, draw);

[x1,y1] = myHistfit(histSamples, 100);
[x2,y2] = myHistfit(randn(1,100000), 100);

figure;
plot(x1, y1,x2,y2);



function [out] = drunkAcc(n, acc, damp, draw)

    out = zeros(1,n);
    
    vel = 0;
    x   = 0;
    
    for i = 1:n
       
        velInc = acc * ((-1 + 2*rand(1,1)) - x * draw);
        
        vel = vel + velInc;
        
        vel = (1-damp) * vel;
        
        xNew = x + vel;
        
        vel = xNew - x;
        x = xNew;
        
        out(i) = x;
    end
end


function [x,y] = myHistfit(bins, ratio)

minVal = min(bins);
maxVal = max(bins);

numBins = length(bins);
numOuts = numBins / ratio;

x = linspace(minVal, maxVal, numOuts);

y = zeros(1,numOuts);

for i = 1:numBins
   [~,idx] = min(abs(bins(i)-x));
   
   y(idx) = y(idx) + 1;
   
end
y = movmean(y,numOuts*0.1)/ratio;



end