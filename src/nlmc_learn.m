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

function Qopt = nlmc_learn(KX,train_labels,r,lambda,Qinit)

% CALL:
%     Qopt = nlmc_learn(KX,train_labels,r,lambda[,Qinit])
% INPUT:
%     KX ~ kernel matrix (p x p)
%     train_labels ~ training_labels (k x m)
%     r ~ number of low-rank components
%     lambda ~ regularization parameter
%     Qinit ~ initial value for Q (p x r)
%     [if Qinit is not provided, randomly initialized]
% OUTPUT:
%     Qopt ~ optimal value for Q (p x r)

    %%% Initial value
    if nargin < 5
        Qinit = rand(size(KX,1),r)-0.5 ;
        [Qinit,~] = qr_gs(Qinit);
    end
    % Vectorize
    qinit = Qinit(:);
    
    %%% Kernel of labels
    KY0 = train_labels'*train_labels;

    %%% OF & Con
    of = @(q) nlmc_objective_function(q,KY0,KX,lambda);
    con = @(q) nlmc_constraint_function(q,r);

    %%% Run KMC
    nlmc_options = optimset(...
        'GradObj','on',...
        'MaxIter',800,...
        'MaxFunEvals',8000,...
        'Display','iter',...
        'GradConstr','on',...
        'DerivativeCheck','off',...
        'FinDiffType','central',...
        'Algorithm','interior-point',...
        'Hessian','fin-diff-grads',...
        'SubproblemAlgorithm','cg');
    [qopt,~,exitflag] = fmincon(of,qinit,[],[],[],[],-ones(size(qinit)),ones(size(qinit)),con,nlmc_options);
    % Check exitflag: if positive fine.
    if exitflag == 0
%         fprintf('nlmc_learn warning: The problem may need more iterations.\n');
        fprintf('nlmc (it) ');
    elseif exitflag < 0
%         fprintf('nlmc_learn warning: There might be a problem with the convergence.\n');
        fprintf('nlmc (co) ');
    end
    
    %%% Reshape
    Qopt = reshape(qopt,size(KX,1),r);
    
end

function [f,g] = nlmc_objective_function(q,KY0,KX,lambda)

    %%% Dimensions
    % Number of training points
    m = size(KY0,1);
    % Number of trainint+testing
    p = size(KX,1);

    %%% Reshape
    r = numel(q)/p;
    Q = reshape(q,p,r);
    Q0 = Q(1:m,:);
    Q1 = Q(m+1:p,:);

    %%% Auxiliar quantities
    % q0'*q0 + lambda*I_r
    aux1 = Q0'*Q0 + lambda*eye(r);
    % kx * q
    aux2 = KX*[Q0;Q1];
    % Q0'*ky0*Q0
    aux3 = Q0'*KY0*Q0;

    %%% Output
    % Cost function
    f = - trace( aux3/aux1 ) ...
        - trace( [Q0' Q1']*aux2 )/(lambda+1);
    % Gradient
    G0 = ((Q0/aux1)*aux3 - KY0*Q0)/aux1 - aux2(1:m,:)/(lambda+1);
    G1 = -aux2(m+1:p,:)/(lambda+1);
    G = [G0; G1];
    g = 2*G(:);
    
end

function [c, c_eq, gc, gc_eq] = nlmc_constraint_function(q,r)
    % Variables
    s = numel(q)/r;
    Q = reshape(q,s,r);
    % Inequalities
    c = [];
    % Equalities
    aux = Q'*Q-eye(r);
    c_eq = aux(:);
    % Gradient intequalities
    gc = [];
    % Gradient equalities
    gc_eq = kron(eye(r),Q);
    aux = [];
    for i = 1:r,
        aux = cat(2,aux,kron(eye(r),Q(:,i)));
    end
    gc_eq = gc_eq + aux;
end
