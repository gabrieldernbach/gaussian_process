clear all; 
close all;


% simple drunk rng algorithm. 
% x[n+1] = x[n] + vel * (rand(-1,+1) - x[n] * draw)
% a approximate normal disribution can be archived with draw ~ 0.2 * vel
% velocity:  determins the maximum per-sample change
% draw:      negative change of velocity based on distance x to 0.

n = 10;
vel  = 0.3;
draw = 0.2 * vel;


samples = [ drunk(n,vel,draw); ...
            drunk(n,vel,draw); ...
            drunk(n,vel,draw); ...
            drunk(n,vel,draw); ...
            drunk(n,vel,draw)];


        
plot(samples');
ylim([-5, +5])

histSamples = drunk(100000, vel, draw);

[x1,y1] = myHistfit(histSamples, 100);
[x2,y2] = myHistfit(randn(1,100000), 100);

figure;
plot(x1, y1,x2,y2, [mean(histSamples), mean(histSamples)],[-10,+10]);

ylim([0, 1])

function [out] = drunk(n, vel, draw)

    out = zeros(1,n);
    
    x   = 0;
    
    for i = 1:n
       
        inc = vel * ((-1 + 2*rand(1,1)) - draw*x);
        
        x = x + inc;
        
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