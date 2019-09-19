clear all; 
close all;

% specify points that the process needs to satisfy
% e.g. start at 0 and end at 0 (we want to loop them)
x = [0,  0.1,  0.2, 0.4, 0.9];
y = [0,    -1,    2,    0.5,  0.2];

 
sigma = 0.01;

[sn,yn, xn, sninv] = genModel(x,y,1000,sigma);


wt = [];
wt2 = [];

for i = (1:8)
    [instance] = instanceWT(yn,sninv);
    wt = [wt;instance];
    
end

figure()
hold on;
plot(wt');
figure()
hold on;
plot(x,y,'o')
plot(xn,yn)
std = 1.96.*sqrt(diag(sn)); % extract 95 percent quantile
plot(xn,yn+std)
plot(xn,yn-std)

%plot(1001* x,y, 'o');

% make a mirrored version that has no discontinuities
wt = [wt,-fliplr(wt)];

% render wavetable, linear sweep throuhg instances, cheap linear interpolation
out = playWT(wt, 50, 44100, 4); % 50Hz, 2 seconds

% generating a coninuously changing wave directly from yn, sn. 
% 50Hz, 2 seconds, new wave every 0.5 seconds
%out = playContinuous(yn, sn, 50, 44100, 10, 0.5); 

%soundsc(out,44100);




