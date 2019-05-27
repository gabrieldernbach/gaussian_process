clear all; 
close all;

% drunk rng based on random change of acceleration
% v[n+1] = (1-damp) * (v[n-1] + acc * (rand(-1,+1) - x[n] * draw));
% x[n+1] = x[n] + v[n]
% ~ normal distribution can be achieved with draw == acc, damp = 0.2

% acceleration: maximum per sample change of acceleration
% damp: damping coefficient of velocity
% draw: negative change of acceleration based on distance x to 0.



acc  = 0.6;
damp = 0.15;
draw = acc;

n = 100;

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
       
        
        % acceleration and velocity update
        vel = vel + acc * ((-1 + 2*rand(1,1)) - x * draw);
        
        % damping
        vel = (1-damp) * vel;
        
        % position update
        x = x + vel;
        
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
y = y ./ (maxVal-minVal);
y = movmean(y,numOuts*0.1)/ratio;



end