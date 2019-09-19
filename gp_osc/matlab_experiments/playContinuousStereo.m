function [out] = playContinuousStereo(yn, sn, f0, fs, t, interval)

outL = playContinuous(yn, sn, f0, fs, t, interval);
outR = playContinuous(yn, sn, f0, fs, t, interval);


out = [outL;outR];
end

