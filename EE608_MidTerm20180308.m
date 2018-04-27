clear;
close all;
clc;
!!!somewhere need to rewrite,matrix and feval
%{
% get the max/min point via KKT algorithm

syms f(x1,x2,x3)
n = 3; % n is the number of variables, the dimension of x

f = x1^2+x2^2+x3^2;
p_f = 'min'; % point flag
c(1) = 5-2*x1-x2; % constrains c means formula>=0, a means formula = 0
c(2) = 2-x1-x3; % constrains
c(3) = x1-1; % constrains
c(4) = x2-2; % constrains
c(5) = x3; % constrains

[m,q] = size(c); % q is the number of inequality constrains, q has no limits
if strcmp(p_f,'max')
    f = -f;
end
mu = sym ('mu',size(c)); % coefficients of inequality
L = f-sum(mu.*c);

d_L_x1 = diff(L,x1); % first order 
d_L_x2 = diff(L,x2);
d_L_x3 = diff(L,x3);
f_mu = mu.*c;

[x1,x2,x3,mu1,mu2,mu3,mu4,mu5] = solve(d_L_x1 == 0,d_L_x2 == 0,d_L_x3 == 0,...
    f_mu == 0,[x1,x2,x3,mu]);
x1 = double(x1);
x2 = double(x2);
x3 = double(x3);
mu1 = double(mu1);
mu2 = double(mu2);
mu3 = double(mu3);
mu4 = double(mu4);
mu5 = double(mu5);
x1_1 = x1(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0);
x2_1 = x2(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0);
x3_1 = x3(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0);

syms x1 x2 x3
% c_1 = subs(c,[x1,x2,x3],[x1_1,x2_1,x3_1]);
c_1 = sym ('c_1',[length(x1_1),length(c)]);
for i = 1:length(x1_1)
    c_1(i,:) = subs(c,[x1,x2,x3],[x1_1(i),x2_1(i),x3_1(i)]);
end
c_2 = double(c_1);
x1_2 = x1_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0);
x2_2 = x2_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0);
x3_2 = x3_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0); % need to optimize!!!
y = double(subs(f,[x1,x2,x3],[x1_2,x2_2,x3_2]));
return
%}

%{
% try to get the optimized the point
y = 100;
for x1 = 1:20
    for x2 = 2:(5-2*x1)
        for x3 = 0:(2-x1)
            if x1^2+x2^2+x3^2<y
                y = x1^2+x2^2+x3^2;
                x = [x1,x2,x3];
            end
        end
    end
end
%}
    
%{
% get the max/min point via KKT algorithm
syms f(xf,xs,xg)
n = 3; % n is the number of variables, the dimension of x

f = 10000*xf+5000*xs+2000*xg;
p_f = 'max'; % point flag
c(1) = xf; % constrains c means formula>=0, a means formula = 0
c(2) = xs;
c(3) = xg;
c(4) = 5-xs;
c(5) = 0.5*(xf+xs+xg)-xg;
c(6) = xf-0.1*(xf+xs+xg);
c(7) = 20000-1000*xf-500*xs-100*xg;

[m,q] = size(c); % q is the number of inequality constrains, q has no limits
if strcmp(p_f,'max')
    f = -f;
end
mu = sym ('mu',size(c)); % coefficients of inequality
L = f-sum(mu.*c);

d_L_xf = diff(L,xf); % first order 
d_L_xs = diff(L,xs);
d_L_xg = diff(L,xg);
f_mu = mu.*c;

[xf,xs,xg,mu1,mu2,mu3,mu4,mu5,mu6,mu7] = solve(d_L_xf == 0,d_L_xs == 0,d_L_xg == 0,...
    f_mu == 0,[xf,xs,xg,mu]);
xf = double(xf);
xs = double(xs);
xg = double(xg);
% mu1 = double(mu1);
% mu2 = double(mu2);
% mu3 = double(mu3);
% mu4 = double(mu4);
% mu5 = double(mu5);
% mu6 = double(mu6);
% mu7 = double(mu7);
xf_1 = xf(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0 & mu6>=0 & mu7>=0);
xs_1 = xs(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0 & mu6>=0 & mu7>=0);
xg_1 = xg(mu1>=0 & mu2>=0 & mu3>=0 & mu4>=0 & mu5>=0 & mu6>=0 & mu7>=0);

syms xf xs xg
% c_1 = subs(c,[x1,x2,x3],[x1_1,x2_1,x3_1]);
c_1 = sym ('c_1',[length(xf_1),length(c)]);
for i = 1:length(xf_1)
    c_1(i,:) = subs(c,[xf,xs,xg],[xf_1(i),xs_1(i),xg_1(i)]);
end
c_2 = double(c_1);
xf_2 = xf_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0 & c_2(:,6)>=0 & c_2(:,7)>=0);
xs_2 = xs_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0 & c_2(:,6)>=0 & c_2(:,7)>=0);
xg_2 = xg_1(c_2(:,1)>=0 & c_2(:,2)>=0 & c_2(:,3)>=0 & c_2(:,4)>=0 & c_2(:,5)>=0 & c_2(:,6)>=0 & c_2(:,7)>=0); % need to optimize!!!
y = double(subs(f,[xf,xs,xg],[xf_2,xs_2,xg_2]));
if strcmp(p_f,'max')
    y = -y;
end
return
%}

%{
% generate the line match the points based on the given function and get the weights
x_s = [-2,-1.8,-1.7,-1.4,-1.1,-0.4,-0.2,0.5,0.9,1.3,1.4,1.6,1.8,2];
y_s = [12.1,9.28,11.63,8.32,1.97,4.52,3.18,1.25,5.37,7.03,5.32,8.62,8.68,10];
N = length(x_s);
k = 1;
syms a b c x;
f = a*x^2+b*x+c;
y_f = subs(f,x,x_s);
f = 1/N*sum((y_s-y_f).^2);
ak = 1;
bk = 1;
ck = 1;
gk(1,1) = double(subs(diff(f,a),[a,b,c],[ak,bk,ck]));
gk(2,1) = double(subs(diff(f,b),[a,b,c],[ak,bk,ck]));
gk(3,1) = double(subs(diff(f,c),[a,b,c],[ak,bk,ck]));
dk = -gk;
H = double(subs(hessian(f,[a,b,c]),[a,b,c],[ak,bk,ck]));
alpha_k = gk.'*gk./(gk.'*H*gk);
err = 2;
while err>1e-3
    ak = ak+alpha_k*dk(1);
    bk = bk+alpha_k*dk(2);
    ck = ck+alpha_k*dk(3);
    err_y(k) = double(subs(f,[a,b,c],[ak,bk,ck]));
    
    gk(1,1) = double(subs(diff(f,a),[a,b,c],[ak,bk,ck]));
    gk(2,1) = double(subs(diff(f,b),[a,b,c],[ak,bk,ck]));
    gk(3,1) = double(subs(diff(f,c),[a,b,c],[ak,bk,ck]));
    dk = -gk;
    alpha_k = gk.'*gk./(gk.'*H*gk);
    err = norm(alpha_k*dk);
        
    k = k+1;
    ff = ak*x_s.^2+bk*x_s+ck;
    figure(1)
    plot(x_s,y_s,'*');hold on;
    plot(x_s,ff);hold on;
    title(['ak=',num2str(ak),';','bk=',num2str(bk),';','ck=',num2str(ck),';','k=',num2str(k)]);hold off;
end
figure(2);
plot(1:k-1,err_y);
xlabel('k'),ylabel('mean squared error');

%}

%{
% simulate the line to match the points
x_y = [1,1;0.5,3;1.5,4;3,3;4,0.5;4,2.5;5.5,2.5;6,1;7,2;5.5,4.5];
x_y = sortrows(x_y);
N = length(x_y);
k = 1;
syms a b c d x;
% f1 = a*x^2+b*x+c+d;
f1 = a*cos(b*x+c)+d;
y_f = subs(f1,x,x_y(:,1));
f = 1/N*sum((x_y(:,2)-y_f).^2);
ak = 1;
bk = 10;
ck = 1;
dk = 1;
gk(1,1) = double(subs(diff(f,a),[a,b,c,d],[ak,bk,ck,dk]));
gk(2,1) = double(subs(diff(f,b),[a,b,c,d],[ak,bk,ck,dk]));
gk(3,1) = double(subs(diff(f,c),[a,b,c,d],[ak,bk,ck,dk]));
gk(4,1) = double(subs(diff(f,d),[a,b,c,d],[ak,bk,ck,dk]));
d_k = -gk;
H = double(subs(hessian(f,[a,b,c,d]),[a,b,c,d],[ak,bk,ck,dk]));
alpha_k = gk.'*gk./(gk.'*H*gk);
err = 2;
while err>1e-2
    ak = ak+alpha_k*d_k(1);
    bk = bk+alpha_k*d_k(2);
    ck = ck+alpha_k*d_k(3);
    dk = dk+alpha_k*d_k(4);
    err_y(k) = double(subs(f,[a,b,c,d],[ak,bk,ck,dk]));
    
    gk(1,1) = double(subs(diff(f,a),[a,b,c,d],[ak,bk,ck,dk]));
    gk(2,1) = double(subs(diff(f,b),[a,b,c,d],[ak,bk,ck,dk]));
    gk(3,1) = double(subs(diff(f,c),[a,b,c,d],[ak,bk,ck,dk]));
    gk(4,1) = double(subs(diff(f,d),[a,b,c,d],[ak,bk,ck,dk]));
    d_k = -gk;
    alpha_k = gk.'*gk./(gk.'*H*gk);
    err = norm(alpha_k*d_k);
        
    k = k+1;
%     ff = ak*x_y(:,1).^2+bk*x_y(:,1)+ck+dk;
    ff = ak*cos(bk*x_y(:,1)+ck)+dk;

    figure(1)
    plot(x_y(:,1),x_y(:,2),'*');hold on;
    plot(x_y(:,1),ff);hold on;
    title(['ak=',num2str(ak),';','bk=',num2str(bk),';','ck=',num2str(ck),';',...
        'dk=',num2str(dk),';','k=',num2str(k)]);hold off;
end
figure(2);
plot(1:k-1,err_y);
xlabel('k'),ylabel('mean squared error');

t = 1:0.01:7;
ff = ak*cos(bk*t+ck)+dk;
% ff = ak*t.^2+bk*t+ck+dk;
figure(3)
plot(x_y(:,1),x_y(:,2),'*');hold on;
plot(t,ff);hold on;
title({['ak=',num2str(ak),';','bk=',num2str(bk),';','ck=',num2str(ck),';',...
    'dk=',num2str(dk),';','k=',num2str(k)],string(f1)});hold off;


%}


    
    
    