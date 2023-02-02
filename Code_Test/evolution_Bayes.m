
function [u ]= evolution_Bayes(Img,u,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf)
u=NeumannBound(u);
K=curvature_central(u); 
H=Heaviside(u,epsilon);
Delta = Dirac(u,epsilon);
KIH = imfilter((H.*Img),Ksigma,'replicate');
KH = imfilter(H,Ksigma,'replicate');
u1= KIH./(KH);
u2 = (KI - KIH)./(KONE - KH);
KI2H = imfilter(Img.^2.*H,Ksigma,'replicate');
sigma1 = (u1.^2.*KH - 2.*u1.*KIH + KI2H)./(KH);
sigma2 = (u2.^2.*KONE - u2.^2.*KH - 2.*u2.*KI + 2.*u2.*KIH + KI2 - KI2H)./(KONE-KH);
sigma1 = sigma1 + eps;
sigma2 = sigma2 + eps;
neighbor = ones(size(Img));
sss = size(Img);
for (i=2:(sss(1)-1))
    for (j=2:(sss(2)-1))
        neighb = ones(3 ,3);
        neighb= Img(i-1:i+1,j-1:j+1);
        sig_neigh = sqrt(var(neighb,0,"all"));
        center_pix = neighb(2,2);
        row_ind= [1 1 1 2 2 3 3 3 ];
        col_ind = [1 2 3 1 3 1 2 3];
        sz=size(neighb);
        lin_in = sub2ind(sz,row_ind,col_ind);
        phi_neigh=u(i-1:i+1,j-1:j+1);
        center_phi=phi_neigh(2,2);
        clique=ones(1 , 9);
        clique(5)=0;
        for k= 1:length(lin_in)
            ind_temp = lin_in(k);
            if sign(center_phi)==sign(phi_neigh(ind_temp))
                clique(ind_temp)=-exp(-abs(center_pix-neighb(ind_temp)));
            else 
                 clique(ind_temp)=(2/(1+exp(-abs(center_pix-neighb(ind_temp)))));
            end
            
        end
        neighbor(i,j)=sum(clique) + log(2*pi)*2*sig_neigh;
        end
end
localForce = (lambda1 - lambda2).*KONE.*log(sqrt(2*pi)) ...
    + imfilter(lambda1.*log(sqrt(sigma1)) - lambda2.*log(sqrt(sigma2)) ...
    +lambda1.*u1.^2./(2.*sigma1) - lambda2.*u2.^2./(2.*sigma2) ,Ksigma,'replicate')...
    + Img.*imfilter(lambda2.*u2./sigma2 - lambda1.*u1./sigma1,Ksigma,'replicate')...
    + Img.^2.*imfilter(lambda1.*1./(2.*sigma1) - lambda2.*1./(2.*sigma2) ,Ksigma,'replicate');
A = -alf.*Delta.*(localForce+neighbor);%data force
P=mu*(4*del2(u) - K);% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
L=nu.*Delta.*K;%length term
u = u+timestep*(L+P+A);
return;
function g = NeumannBound(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
function K = curvature_central(u);
[bdx,bdy]=gradient(u);
mag_bg=sqrt(bdx.^2+bdy.^2)+1e-10;
nx=bdx./mag_bg;
ny=bdy./mag_bg;
[nxx,nxy]=gradient(nx);
[nyx,nyy]=gradient(ny);
K=nxx+nyy;
function h = Heaviside(x,epsilon)
h=0.5*(1+(2/pi)*atan(x./epsilon));
function f = Dirac(x, epsilon)
f=(epsilon/pi)./(epsilon^2.+x.^2);
