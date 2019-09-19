function [wt] = instanceWT(yn,sninv)
    %wt = mvnrnd(yn,sn);

    n = length(yn);
    smooth = 5; % fade over 400 samples
    
    in = (1:1:smooth)./smooth;
    out = (smooth:-1:1)./smooth;
    hold = ones(length(yn)-2*smooth,1);
    fade = [in';hold;out'];
    
    var = sninv * randn(n,1);
    varfaded = var .* fade;
    
    wt = yn + varfaded;
    wt = wt';
        
    
    wt = wt(:,(1:(end-1)));    
    %wt = preprocessWTPolynom(wt);
end

