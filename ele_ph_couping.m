% QMC for electron-phonon coupling with 1D/2D
clear;
tic;
warning('off');


Lx = 4;
Ly = 4;

if Ly == 1
    dim = 1;
else
    dim = 2;
end

len = Lx*Ly*dim;
beta = 1;
dt = 0.1; % d_tau, step of imaginary time
num_w = 10;
num_m = 20;
N = 1; % # of fermion flavors
M = beta/dt;
tl0 = 0.0001; % time step for Lengevin equation
Fmax = 100;
m = 2; % phonon mass
k = 2; % elestric constant
g = 1; % el-ph couping constant
t = 1; % hoping constant for el
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

Kb = cell(len,1);
for b = 1:len
    Kb{b} = gen_Kb(b);
end


field_p = randn(len,M);

pos_p = zeros(num_m,1);

num_store = zeros(num_m,len,M);

G = cell(len,M);
G_final = cell(len,M);
for p = 1:M
    G_final{p} = zeros(Lx*Ly);
end

tl_store = zeros(num_m,1);
tl = tl0;

% warm-up
for n = 1:num_w
    field_l = zeros(len,M);
    field_r = zeros(len,M);
    field_l(:,1:M-1) = field_p(:,2:end);
    field_r(:,2:end) = field_p(:,1:M-1);
    field_l(:,end) = field_p(:,1);
    field_r(:,1) = field_p(:,end);

    temp_tr = zeros(len,M);
    for b = 1:len
        KKb = Kb{b};
        for p = 1:M
            temp1 = gen_H(field_p,1,b-1,1,p-1,Lx,Ly,g,t,dim);
            temp2 = gen_H(field_p,1,b-1,p,M,Lx,Ly,g,t,dim);
            temp3 = gen_H(field_p,b,len,1,p-1,Lx,Ly,g,t,dim);
            temp4 = gen_H(field_p,b,len,p,M,Lx,Ly,g,t,dim);
            B1 = expm(dt*temp1);
            B2 = expm(dt*temp2);
            B3 = expm(dt*temp3);
            B4 = expm(dt*temp4);
            G{b,p} = inv(eye(Lx*Ly) + B1*B2*B3*B4);
            temp_tr(b,p) = trace(KKb*(eye(Lx*Ly)-G{b,p}));
        end
    end
    
    fact = dt*k*field_p + m/dt * (2*field_p - field_l - field_r) + N*g*dt*temp_tr;
%     field_p = field_p - MM*tl*fact + sqrt(2*tl)*MM_s*randn(len_p,1);

    tl = tl*Fmax/max(fact,[],"all");

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

    temp_tr = zeros(len,M);
    for b = 1:len
        KKb = Kb{b};
        for p = 1:M
            temp1 = gen_H(field_p,1,b-1,1,p-1,Lx,Ly,g,t,dim);
            temp2 = gen_H(field_p,1,b-1,p,M,Lx,Ly,g,t,dim);
            temp3 = gen_H(field_p,b,len,1,p-1,Lx,Ly,g,t,dim);
            temp4 = gen_H(field_p,b,len,p,M,Lx,Ly,g,t,dim);
            B1 = expm(dt*temp1);
            B2 = expm(dt*temp2);
            B3 = expm(dt*temp3);
            B4 = expm(dt*temp4);
            G{b,p} = inv(eye(Lx*Ly) + B1*B2*B3*B4);
            temp_tr(b,p) = trace(KKb*(eye(Lx*Ly)-G{b,p}));
        end
    end
    
    fact = dt*k*field_p + m/dt * (2*field_p - field_l - field_r) + N*g*dt*temp_tr;
%     field_p = field_p - MM*tl*fact + sqrt(2*tl)*MM_s*randn(len_p,1);

    tl = tl*Fmax/max(fact,[],"all");
    tl_store(n) = tl;

    temp1 = tl*MM.*my_F(fact,M);
    temp2 = sqrt(2*tl*MM).*my_F(randn(len,M),M);
    field_p = field_p - real(my_invF(temp1-temp2,M));

    pos_p(n) = mean(field_p,"all");
    num_store(n,:,:) = field_p;

    for p = 1:M
        temp_l = gen_H(field_p,1,len,1,p-1,Lx,Ly,g,t,dim);
        temp_r = gen_H(field_p,1,len,1,p-1,Lx,Ly,g,t,dim);
        B_l = expm(dt*temp_l);
        B_r = expm(dt*temp_r);
        G_final{p} = G_final{p} + G{p}/num_m;
    end
end

mean_pos_p = mean(pos_p.*tl_store)/sum(tl_store)
% histogram(num_store)

toc;

% for fft
function y = my_F(x,len_p)
seq = -len_p/2+1:1:len_p/2;
vn = seq/len_p;
tau = 1:len_p;
y = nufft(x,tau,-vn,2)/len_p;
end

% for inv_fft
function y = my_invF(x,len_p)
seq = -len_p/2+1:1:len_p/2;
vn = seq/len_p;
tau = 1:len_p;
y = nufft(x,vn,tau,2);
end

function Tij = gen_H(field_p,b1,b2,t1,t2,Lx,Ly,g,t,dim)
len = Lx*Ly;
Tij = zeros(len);
count = 0;

if b2 < 1 || t2 < 1
    Tij = eye(len);
    return
end

if dim == 1
    q = field_p(:,t1:t2);
    count = 0;
    for i = b1:min([Lx-1,b2])
        temp = t - g*sum(q(i,:));
        Tij(i,i+1) = Tij(i,i+1) + temp;
        Tij(i+1,i) = Tij(i+1,i) + temp;
        count = count +1;
    end
    if b2 == Lx
        temp = t - g*sum(q(Lx,:));
        Tij(Lx,1) = Tij(Lx,1) + temp;
        Tij(1,Lx) = Tij(1,Lx) + temp;
        count = count +1;
    end

else
%     q_t = field_p(1:len,t1:t2);
%     q_v = field_p(len+1:end,t1:t2);
    for i = 1:Ly-1
        for j = 1:Lx-1
            pos = (i-1)*Lx+j;

            if pos >= b1 && pos <=b2
                temp_t = t - g*sum(field_p(pos,:));
                Tij(pos,pos+1) = Tij(pos,pos+1) + temp_t;
                Tij(pos+1,pos) = Tij(pos+1,pos) + temp_t;
            end

            if pos+len >= b1 && pos+len <=b2
                temp_v = t - g*sum(field_p(pos+len,:));
                Tij(pos,pos+Lx) = Tij(pos,pos+Lx) + temp_v;
                Tij(pos+Lx,pos) = Tij(pos+Lx,pos) + temp_v;
            end
            count = count +1;
        end

        pos = i*Lx;
        if pos >= b1 && pos <=b2
            temp_t = t - g*sum(field_p(pos,:));
            Tij(pos,pos-Lx+1) = Tij(pos,pos-Lx+1) + temp_t;
            Tij(pos-Lx+1,pos) = Tij(pos-Lx+1,pos) + temp_t;
        end
        if pos+len >= b1 && pos+len <=b2
            temp_v = t - g*sum(field_p(pos+len,:));
            Tij(pos,pos+Lx) = Tij(pos,pos+Lx) + temp_v;
            Tij(pos+Lx,pos) = Tij(pos+Lx,pos) + temp_v;
        end
        count = count +1;

    end
    for j = 1:Lx-1
        pos = (Ly-1)*Lx+j;
        if pos >= b1 && pos <=b2
            temp_t = t - g*sum(field_p(pos,:));
            Tij(pos,pos+1) = Tij(pos,pos+1) + temp_t;
            Tij(pos+1,pos) = Tij(pos+1,pos) + temp_t;
        end
        if pos+len >= b1 && pos+len <=b2
            temp_v = t - g*sum(field_p(pos+len,:));
            Tij(pos,pos+Lx-len) = Tij(pos,pos+Lx-len) + temp_v;
            Tij(pos+Lx-len,pos) = Tij(pos+Lx-len,pos) + temp_v;
        end
        count = count +1;
    end
    if len <= b2
        temp_t = t - g*sum(field_p(len,:));
        Tij(len,len-Lx+1) = Tij(len,len-Lx+1) + temp_t;
        Tij(len-Lx+1,len) = Tij(len-Lx+1,len) + temp_t;
    end
    if 2*len == b2
        temp_v = t - g*sum(field_p(2*len,:));
        Tij(len,Lx) = Tij(len,Lx) + temp_v;
        Tij(Lx,len) = Tij(Lx,len) + temp_v;
    end
    count = count +1;
end

end

function Kb = gen_Kb(b)
global Lx Ly dim
len = Lx*Ly;
Kb = sparse(zeros(len));

if dim == 1
    count = 0;
    if b ~= len
        i = b;
        Kb(i,i+1) = Kb(i,i+1) + 1;
        Kb(i+1,i) = Kb(i+1,i) + 1;
    else
        Kb(Lx,1) = Kb(Lx,1) + 1;
        Kb(1,Lx) = Kb(1,Lx) + 1;
    end

else
    if b <= len
        if mod(b,Lx) ~= 0
            Kb(b,b+1) = Kb(b,b+1) + 1;
            Kb(b+1,b) = Kb(b+1,b) + 1;
        else
            Kb(b,b-Lx+1) = Kb(b,b-Lx+1) + 1;
            Kb(b-Lx+1,b) = Kb(b-Lx+1,b) + 1;
        end
    else
        b = b -len;
        if b <= len-Lx

            Kb(b,b+Lx) = Kb(b,b+Lx) + 1;
            Kb(b+Lx,b) = Kb(b+Lx,b) + 1;
        else
            Kb(b,b+Lx-len) = Kb(b,b+Lx-len) + 1;
            Kb(b+Lx-len,b) = Kb(b+Lx-len,b) + 1;
        end
    end
end

end
