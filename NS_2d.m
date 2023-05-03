clear all; close all; clc;

% define time stuff
tinit = 0;
tfinal = 1;
dt = .1;
kmax = ceil((tfinal-tinit)/dt);

% define more stuff
Re = 1;

% define grid (uniform only for now)
m = 3;
xmin = -pi/2;
xmax = pi/2;
xlen = linspace(xmin,xmax,m+1);
dx = (xmax-xmin)/m;
[X,Y] = meshgrid(xlen,xlen);

% initialize
u = zeros(m-1,m); 
v = zeros(m,m-1);

utrue = zeros(m-1,m);
vtrue = zeros(m,m-1);

ustar = u;
vstar = v;

uW = u; uE = u;
uN = u; uS = u;

vW = v; vE = v;
vN = v; vS = v;

ubc = u; vbc = v;

usave = zeros(m-1,m,kmax);
vsave = zeros(m,m-1,kmax);

u_norm = zeros(kmax,1);
v_norm = zeros(kmax,1);

% Construct the operators (since the mesh doesn't change)
Ap = spdiags([-1 1 0;ones(m-2,1)*[-1 2 -1];0 1 -1],-1:1,m,m)';
Lp = (1/dx^2)*(kron(speye(m),Ap)+kron(Ap,speye(m)));
Lp(m*m,m*m) = Lp(m*m,m*m)+1;

Lu = speye((m-1)*m)+dt/Re*(kron(speye(m),K1(m-1,dx,2))+...
     kron(K1(m,dx,3),speye(m-1)));

Lv = speye(m*(m-1))+dt/Re*(kron(speye(m-1),K1(m,dx,3))+...
     kron(K1(m-1,dx,2),speye(m)));

for k = 1:kmax

    % compute ustar, vstar, utrue, vtrue
    for i = 1:m
        for j = 1:m
            if i < m
                ustar(i,j) = u(i,j) - dt*getAu(dt*k,Re,X(i,j+1)+dx/2,Y(i,j+1));
                utrue(i,j) = uexact(dt*k,Re,X(i,j+1)+dx/2,Y(i,j+1));
            end

            if j < m
                vstar(i,j) = v(i,j) - dt*getAv(dt*k,Re,X(i+1,j),Y(i+1,j)+dx/2);
                vtrue(i,j) = vexact(dt*k,Re,X(i+1,j),Y(i+1,j)+dx/2);
            end
        end
    end

    % compute updated bcs
    for i = 1:m
        if i < m
            uW(i,1) = uexact(dt*k,Re,X(i,1),Y(i+1,1));
            uE(i,m) = uexact(dt*k,Re,X(i,m),Y(i+1,m));

            vS(1,i) = vexact(dt*k,Re,X(1,i+1),Y(1,i));
            vN(m,i) = vexact(dt*k,Re,X(m,i+1),Y(m,i));
        end

        uS(1,i) = uexact(dt*k,Re,X(1,i)+dx/2,Y(1,i));      % using exact val
        uN(m-1,i) = uexact(dt*k,Re,X(m-1,i)+dx/2,Y(m,i));  % using exact val

        vW(i,1) = vexact(dt*k,Re,X(i,1),Y(i,1)+dx/2);
        vE(i,m-1) = vexact(dt*k,Re,X(i,m),Y(i,m)+dx/2);
    end

    ubc = dt/Re*(ustar+2*uS+2*uN+uE+uW)/dx^2;
    vbc = dt/Re*(vstar+vS+vN+2*vE+2*vW)/dx^2;
    

    % solve implicit viscosity
    rhs_u = reshape(ustar+ubc,[],1);
    rhs_v = reshape(vstar+vbc,[],1);

    usol = Lu\rhs_u;
    vsol = Lv\rhs_v;

    ustar = reshape(usol,m-1,m);
    vstar = reshape(vsol,m,m-1);

    % compute Hodge (p)
    rhs_p = reshape((diff([uS(1,:);ustar;uN(m-1,:)]) + diff([vW(:,1),vstar,vE(:,m-1)]'))/dx,[],1);
    rhs_p1 = reshape((diff([uS(1,:);utrue;uN(m-1,:)]) + diff([vW(:,1),vtrue,vE(:,m-1)]'))/dx,[],1);

    psol = Lp\rhs_p;
    phi = reshape(psol,m,m);

    % update u and v
    u = ustar - diff(phi)/dx;
    v = vstar - diff(phi)'/dx;

    u_norm(k) = norm(u-utrue);
    v_norm(k) = norm(v-vtrue);

    usave(:,:,k) = u;
    vsave(:,:,k) = v;

    fprintf('Time : %d\nNorm u: %d\nNorm v: %d\n\n',k*dt,u_norm(k),v_norm(k));
    
end


function [Au] = getAu(t,Re,x,y)

    u  = -cos(x)*sin(y)*exp(-2*t/Re);
    v  =  sin(x)*cos(y)*exp(-2*t/Re);
    ux =  sin(x)*sin(y)*exp(-2*t/Re);
    uy = -cos(x)*cos(y)*exp(-2*t/Re);

    Au = u*ux + v*uy;
end

function [Av] = getAv(t,Re,x,y)

    u  = -cos(x)*sin(y)*exp(-2*t/Re);
    v  =  sin(x)*cos(y)*exp(-2*t/Re);
    vx =  cos(x)*cos(y)*exp(-2*t/Re);
    vy = -sin(x)*sin(y)*exp(-2*t/Re);

    Av = u*vx + v*vy;

end

function [ue] = uexact(t,Re,x,y)
    ue  = -cos(x)*sin(y)*exp(-2*t/Re);
end

function [ve] = vexact(t,Re,x,y)
    ve  =  sin(x)*cos(y)*exp(-2*t/Re);
end

function A = K1(n,h,bcn)
% a11: Neumann=1, Dirichlet=2, Dirichlet mid=3;
    A = spdiags([-1 bcn 0;ones(n-2,1)*[-1 2 -1];0 bcn -1],-1:1,n,n)'/h^2;
end