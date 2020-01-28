function hx = NFC(S_H,T_H,options,ker)
% hx: dxn hidden representation
SrcSamp = size(S_H,2);
TarSamp = size(T_H,2);

noise = options.noises;
lambda = options.lambda;
beta = options.beta;
disp('MMD...');
M11 = (1/SrcSamp^2)*ones(SrcSamp,SrcSamp);
M12 = -1/(SrcSamp*TarSamp)*ones(SrcSamp,TarSamp);
M21 =M12';
M22 = (1/TarSamp^2)*ones(TarSamp,TarSamp);
M = [M11,M12;M21,M22];
xx = [S_H,T_H];
% kk = kernelmatrix(ker,xx,xx);
 options.ker = 'linear';     % kernel: 'linear' | 'rbf' | 'lap'
        options.eta = 2.0;          % eigenspectrum damping factor
kk = TKL(S_H, T_H,options);
[d,~] = size(kk);
kkb =kk;
       
KS_H = [(1:SrcSamp)', kk(1:SrcSamp, 1:SrcSamp)];
KT_H =[(1:TarSamp)', kk(SrcSamp+1:end, 1:SrcSamp)];
options.L = LaplacianMatrix(KS_H',KT_H',10);
 
%% corruption vector
q = ones(d, 1)*(1-noise);
Q0 = kkb*kkb';
Q1 = Q0.*(q*q');
Q1(1:d+1:end) = q.*diag(Q0);
SMMD = kkb*M*kkb';
SMMD2 = kkb*diag(diag(M))*kkb';
Q2 = SMMD.*(q*q');
Q2(1:d+1:end) = q.*q.*diag(SMMD)+q.*(1-q).*diag(SMMD2);

disp('Manifold...');
% S_H = kkb(:,1:options.size);
% T_H = kkb(:,options.size+1:end);
% 
% L = LaplacianMatrix(S_H,T_H,10);
L = options.L;
SManifold = kkb*L*kkb';
SManifold2 = kkb*diag(diag(L))*kkb';
Q3 = SManifold.*(q*q');
Q3(1:d+1:end) = q.*q.*diag(SManifold)+q.*(1-q).*diag(SManifold2);

P =(1-noise)*kkb*xx';
reg = 0.0001*eye(d);
W = P'/(Q1+lambda*Q2+reg+beta*Q3);
hx = tanh(W*kkb);


