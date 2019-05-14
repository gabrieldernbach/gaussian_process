clear all;
close all;

x = [0,  0.1,  0.2, 0.4, 0.9];
y = [0,    -1,    2,    0.5,  1];

sigma = 0.05;

[sn,yn, xn, sninv] = genModel(x,y,1001,sigma);

t = 15; %seconds
fs = 44100;

% 50Hz straight forward
out1 = playContinuousStereo(yn, sninv, [50,50,50,50,50], fs, t, 10);

% 100Hz, modulation from 10s to 0.05s
out2 = linspace(1,4,t*fs) .* playContinuousStereo(yn, sninv, 100, fs, t, [10, 0.2, 0.05,10]);

% 175Hz zo 155hz to 150hz 
out3 = 0.5 * playContinuousStereo(yn, sninv, [150,150,150,150,150], fs, t, 1);


chord = out1 + out2 + out3;