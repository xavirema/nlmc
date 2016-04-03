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

function [Q,R] = qr_gs(A)
%
% CALL:
%     [Q,R] = qr_gs(A)
% Compute the QR decomposition of A using  Gram-Schmidt
%
    [m,n] = size(A);
    % compute QR using Gram-Schmidt
    Q = zeros(m,n);
    R = zeros(n);
    for j = 1:n
       v = A(:,j);
       for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
       end
       R(j,j) = norm(v);
       Q(:,j) = v/R(j,j);
    end
end