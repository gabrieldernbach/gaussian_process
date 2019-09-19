function [wt] = preprocessWT(wt)

% smoothing discontinuity at 0 by inserting a polynom of 3rd order 
% taking position and derivate (averaged across two points) into account 

N = size(wt,2);

%overlap 

overlap = 0.1;
overlapInt = ceil(N*overlap);

x0  = -overlapInt;
y0  = wt(:,N-overlapInt+1);
dy0 = (wt(:,N-overlapInt+1) - wt(:,N-overlapInt-1))*overlapInt/2;

x1  = overlapInt;
y1  = wt(:,overlapInt+1);
dy1 = (wt(:,overlapInt+2) - wt(:,overlapInt))*overlapInt/2;

a = (1/4)*dy0+(1/4)*y0+(1/4)*dy1-(1/4)*y1; 
b = -(1/4)*dy0+(1/4)*dy1;
c = -(1/4)*dy0-(3/4)*y0-(1/4)*dy1+(3/4)*y1;
d = (1/4)*dy0+(1/2)*y0-(1/4)*dy1+(1/2)*y1;

x = linspace(-1,+1,2*overlapInt+1);

y = a.*x.^3 + b.*x.^2 + c.*x + d;

wt(:,(N-overlapInt+1):(end)) = y(1:overlapInt);
wt(:,1:overlapInt+1)         = y(overlapInt+1:end);




end