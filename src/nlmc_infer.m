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

function [e_test_labels, e_train_labels] = nlmc_infer(Qopt,train_labels,r,lambda)
% 
% CALL:
%     [e_test_labels, e_train_labels] = nlmc_infer(Qopt,train_labels,r,lambda)
% INPUT:
%     Qopt ~ optimal value of Q (p x r)
%     train_labels ~ the training  labels (k x m)
%     r ~ number of low-rank components
%     lambda ~ regularization parameter
% OUTPUT:
%     e_test_labels ~ estimated test labels (k x n)
%     e_train_labels ~ estimated train labels (k x m)
%     
    
    %%% Infer Y1 = L0*Q1' adn Y0 = L0*Q0'
    nTrain = size(train_labels,2);
    % Decompose optimal Q into Q0 and Q1
    Q0 = Qopt(1:nTrain,:);
    Q1 = Qopt(nTrain+1:end,:);
    % Estimate L0
    L0 = train_labels*Q0/(Q0'*Q0 + lambda*eye(r));
    % Estimate labels
    e_test_labels = L0*Q1';
    e_train_labels = L0*Q0';
end
