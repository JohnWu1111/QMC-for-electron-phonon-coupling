% QMC for electron-phonon coupling, this program is for demo and only
% include phonon part
clear;
tic;

Lx = 10;
Ly = 1;
len = Lx*Ly;
beta = 5;
dt = 0.1;
num_w = 200;
num_m = 1000;
M = beta/dt;
tl = 0.001;
m = 1;
k = 0.16;
omega_0 = sqrt(k/m);
len_p = len*M;

% construction of matrix "M"
% MM = diag(rand(len_p,1));
% [Q,R] = qr(rand(len_p));
% MM = Q'*MM*Q;
% e = eig(MM);
% MM_s = MM^1/2;
seq = -M/2+1:1:M/2;
vn = seq/M;
MM = (dt*k + 4*m/dt)./(dt*k + m/dt*(2 - 2*cos(2*pi*vn)));
% MM = ones(len_p,1);


field_p = randn(len,M);

pos_p = zeros(num_m,1);

num_store = zeros(num_m,len,M);


% warm-up
for n = 1:num_w
    field_l = zeros(len,M);
    field_r = zeros(len,M);
    field_l(:,1:M-1) = field_p(:,2:end);
    field_r(:,2:end) = field_p(:,1:M-1);
    field_l(:,end) = field_p(:,1);
    field_r(:,1) = field_p(:,end);
    
    fact = dt*k*field_p + m/dt * (2*field_p - field_l - field_r);
%     field_p = field_p - MM*tl*fact + sqrt(2*tl)*MM_s*randn(len_p,1);

    temp1 = tl*MM.*my_F(fact,M);
    temp2 = sqrt(2*tl*MM).*my_F(randn(len,M),M);
    field_p = field_p - real(my_invF(temp1-temp2,M));
end

% measurement
for n = 1:num_m
    field_l = zeros(len,M);
    field_r = zeros(len,M);
    field_l(:,1:M-1) = field_p(:,2:end);
    field_r(:,2:end) = field_p(:,1:M-1);
    field_l(:,end) = field_p(:,1);
    field_r(:,1) = field_p(:,end);
    
    fact = dt*k*field_p + m/dt * (2*field_p - field_l - field_r);
%     field_p = field_p - MM*tl*fact + sqrt(2*tl)*MM_s*randn(len_p,1);

    temp1 = tl*MM.*my_F(fact,M);
    temp2 = sqrt(2*tl*MM).*my_F(randn(len,M),M);
    field_p = field_p - real(my_invF(temp1-temp2,M));

    pos_p(n) = mean(field_p,"all");
    num_store(n,:,:) = field_p;
end

mean_pos_p = mean(pos_p)
histogram(num_store)

toc;

function y = my_F(x,len_p)
seq = -len_p/2+1:1:len_p/2;
vn = seq/len_p;
tau = 1:len_p;
y = nufft(x,tau,-vn,2)/len_p;
end

function y = my_invF(x,len_p)
seq = -len_p/2+1:1:len_p/2;
vn = seq/len_p;
tau = 1:len_p;
y = nufft(x,vn,tau,2);
end
