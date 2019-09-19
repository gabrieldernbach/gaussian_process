function [wt] = preprocessWT(wt)

% smoothing by crossfade, don't use, it doesn't work

N = size(wt,2);

%overlap 

overlap = 0.05;
overlapInt = ceil(N*overlap);


fIn  = wt(:,(1:overlapInt));
fOut = wt(:,(N-overlapInt+2):(N));
fOut = [fOut, wt(:,1)];
    
fIn  = [ -fliplr(fIn), fIn(:,2:(end))];
fOut = [fOut(:,1:(end-1)), -fliplr(fOut)];

% crossfade
fade = linspace(0,1,size(fIn,2));
fadeFull = fIn .* fade + (1-fade) .* fOut;

%insert in wt
wt(:,(N-overlapInt+2):N) = fadeFull(:,1:(overlapInt-1));
wt(:,(1:overlapInt))     = fadeFull(:,(end-overlapInt+1):end);


end