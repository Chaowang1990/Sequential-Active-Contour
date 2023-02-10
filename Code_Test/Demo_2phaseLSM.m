clc;clear all;close all;
%------------------------------------
% This Matlab program demomstrates the level set algorithm in paper:
%   Kaihua Zhang, Lei Zhang,et al "Locally Statistical Level Set Method for Image Segmentation with Intensity Inhomogeneity Locally Statistical Level Set Method for Image Segmentation with Intensity Inhomogeneity" 
%   submitted, 2011
% Author: Kaihua Zhang
% E-mail: cskhzhang@comp.polyu.edu.hk, cslzhang@comp.polyu.edu.hk  
% http://www4.comp.polyu.edu.hk/~cskhzhang/
% http://www4.comp.polyu.edu.hk/~cslzhang/
%  Notes:
%   1. Some parameters may need to be modified for different types of images. Please contact the author if any problem regarding the choice of parameters.
%   2. Intial contour should be set properly.
% Date: 20/7/2011
%------------------------------------Set path
addpath('./image');
%------------------------------------
flag =1;% Utilize different flags to demonstrate different experiments.
%------------------------------------
switch flag
      case 1
          I = imread('noisyNonUniform.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 20;
          sigma = 2.5;
          Iternum = 300; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.1;          
       case 2
          I = imread('nonHom3.gif'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 20;
          sigma = 6;
          Iternum = 60; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.1;
       case 3
          I = imread('vessel2.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = 80;
          jc = 60;
          r  = 10;
          sigma = 2.5;
          Iternum = 720; %Iterations
          epsilon = 1;
          timestep = 1; 
          delt2 = 0.1;
       case 4
          I = imread('vessel3.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = 80;
          jc = 90;
          r  = 20;
          sigma = 6;
          Iternum = 300; %Iterations
          epsilon = 1;
          timestep = 1; 
          delt2 = 0.1;
       case 5
          I = imread('f95.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 20;
          sigma = 2.5;
          Iternum = 700; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.1;
       case 6
          I = imread('T2.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 20;
          sigma = 10;
          Iternum = 300; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.1;
        case 7
          I = imread('3Tb.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 15;
          sigma = 2.5;
          Iternum = 600; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.1;
         case 8
          I = imread('5.bmp'); % for 3t MRI
          I = double(I(:,:,1));
          [nrow,ncol] = size(I);
          ic = nrow/2;
          jc = ncol/2;
          r  = 15;
          sigma = 5;
          Iternum = 400; %Iterations
          epsilon = 1;
          timestep = 1;
          delt2 = 0.01;
end

K = ones(4*sigma+1);%The constant kerenl
%K=fspecial('gaussian',round(2*sigma)*2+1,sigma);% The Gaussian Kernel
%initialization
dim = 2; % two phase model
%--------------------------------------------------------------------------
figure;imagesc(I, [0, 255]);colormap(gray);hold on; axis off;
phi = sdf2circle(nrow,ncol, ic,jc,r);
hold on;
[c,h] = contour(phi,[0 0],'r');
%--------------------------------------------------------------------------
%-----Parameter initialization
b(1:nrow,1:ncol) = 1;
for i = 1:dim
s(1:nrow,1:ncol,i) = i;    
end
%---------------------------------
numframes= Iternum+1; 
A=moviein(Iternum/20+1); % create the movie matrix 
A(:,1)=getframe(gcf); 
count = 2;
%---------------------------------
for i = 1:Iternum
    u = compute_u(phi,epsilon);% the membership function, i.e., u(:,:,1)=H(phi),u(:,:,2)=1-H(phi) as in Eq.(14)
    c = compute_c(I,K,u,b);% ci in Eq.(16)
    b = compute_b(I,K,u,c,s);% b in Eq.(17)
    s = compute_s(I,b,K,c,u);% the variance of corresponding region. see the sigma in Eq.(18)
    d = computer_d(I,K,s,b,c);
    phi = TwoPhLSE(phi,d,timestep,epsilon);%Two Phase Level Set Evolution
    phi = phi + delt2*4*del2(phi);% Level set regularization
    %-----------------------------------
    if mod(i,20)==0
    imagesc(I,[0 255]);colormap(gray)
    hold on;
    contour(phi,[0 0],'r','LineWidth',2);
    iterNum=[num2str(i), ' iterations'];  
    title(iterNum);hold off; 
    pause(0.01);
    A(:,count)=getframe(gcf);
    count = count+1;
    end
    
end
movie2avi(A,'movie','compression','none');