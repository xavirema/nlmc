%     nlmc implements Non-Linear Matrix Completion
%     Copyright (C) 2016 Xavier Alameda-Pineda [xavi.alameda@gmail.com]
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
%     Please cite the following article if you use this software:
%     
%     X. Alameda-Pineda, Y. Yan, E. Ricci and N. Sebe, 
%     Recognizing Emotions from Abstract Paintings using Non-Linear Matrix Completion
%     IEEE Computer Vision and Pattern Recognition 2016.
%
%     This particular function was adopted from www.kernel-methods.net
%     More precisely from:  Tijl De Bie, february 2003. Adapted: october 2004 (for speedup).

function K = rbf(coord,sig)

%function K = rbf(coord,sig)
%
% Computes an rbf kernel matrix from the input coordinates
%
%INPUTS
% coord =  a matrix containing all samples as rows
% sig = sigma, the kernel width; squared distances are divided by
%       squared sig in the exponent
%
%OUTPUTS
% K = the rbf kernel matrix ( = exp(-1/(2*sigma^2)*(coord*coord')^2) )
%
%
% For more info, see www.kernel-methods.net

    n=size(coord,1);
    K=coord*coord'/sig^2;
    d=diag(K);
    K=K-ones(n,1)*d'/2;
    K=K-d*ones(1,n)/2;
    K=exp(K);
    
end