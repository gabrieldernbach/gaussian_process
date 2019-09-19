function [sn,yn, xn, sninv] = genModel(x,y, n, sigma)

	% assume points in the domain 0 <= x < 1 strictly!
    x = [x-1, x, x+1];
    y = [y, y, y];
    
    xn = linspace(0,1,n);

    kk   = kernel(x,x, sigma); % auto covariance of training
    kkn  = kernel(xn,x, sigma); % cross covariance with evaluation
    knkn = kernel(xn,xn, sigma); % auto covariance of evaluation

    yn = kkn' / kk * y'; % target mean prediction
    sn = knkn - kkn' / kk * kkn; % target variance prediction

    [U,S,V] = svd(sn);
    A = U * sqrt(S);
    
    sninv = A;

end



function out = kernel(x,y,sig) % define a kernel as similarity measure
    out = exp(-(x-y').^2 ./ (2*sig^2));
end
