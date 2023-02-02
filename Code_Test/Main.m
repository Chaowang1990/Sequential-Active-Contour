clear all

% Change the line below to work directory
cd '\\iowa.uiowa.edu\shared\engineering\home\afallahdizcheh\windowsdata\Desktop\Image_segmentatation';

image = dicomread('7.dcm');
imshow(image);


%imshow(img)
%impixelinfo
x = [471 547 620 947 547 145];
y = [133 157 133 599 724 599];

bw = poly2mask(x,y,768,1024);
%imshow(bw)



[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));

newThreshold = 0.2;


bin = imbinarize(image,newThreshold);
%imshow(bin);


maskedImage = image; % Initialize
maskedImage(~bw) = 0; % Erase background.

%imshow(maskedImage);

%new_image = ~maskedImage;
%imshow(new_image)

image = maskedImage;


[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));
% No black pixels here
newThreshold =0.1;
% finding th threshold needs to be updated

bin = imbinarize(image,newThreshold);
%imshow(bin);

binaryImage = imclearborder(bin);
%imshow(binaryImage);

%imagen = image.*bin;
%imshow(imagen);

%[B,L] = bwboundaries(bin,'holes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
%hold on
%for k = 1:length(B)
%   boundary = B{k};
%   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%end


binImage = imfill(bin,'holes');
%imshow(binImage);


binImage(~bw)=1;
binImage = ~binImage;
%imshow(binImage);
%now we extract the white area, we can select multiple ones
whitearea = bwareafilt(binImage,1);
%imshow(whitearea);


%check_point = find(whitearea);
%check_point_t = check_point(floor(length(check_point/2)));
%ismember(check_point_t,find(whitearea))
%
%test_tresh = 0.1;
%for i = 0:15
%    newThreshold = test_tresh+(i)*0.02;
%bin = imbinarize(image,'adaptive',newThreshold);
%binaryImage = imclearborder(bin);
%binImage = imfill(bin,'holes');
%binImage(~bw)=1;
%binImage = ~binImage;
%whitearea = bwareafilt(binImage,1);
%test = find(whitearea);
%res= ismember(check_point_t,test);
%if res == 0
%    final_thresh = test_tresh+(i-1)*0.02;
%    final_white_area =  whitearea;
%   break 
%end
%if i ==10
%    final_thresh = test_tresh+(i)*0.02;
%    final_white_area =  whitearea;
%end
%end



boundries = bwboundaries(whitearea);

[B,L] = bwboundaries(whitearea,'noholes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
stats = regionprops(whitearea,'area', 'Centroid','BoundingBox');
centre=stats.Centroid;
%%%%%%%%Resizing maybe?
Im = whitearea;
[rows,columns] = size(Im);
%imshow(Im)

%extract properties of the region


%%%extending using centroid

%extractin polygon containing area

boxbw = stats.BoundingBox;
boxbw(1:2) = boxbw(1:2) - 90;
boxbw(3:4) = boxbw(3:4) + 140;
%image = dicomread('4.dcm');
%imshow(maskedImage);
%maskedImage=insertShape(maskedImage, 'Line', [470, 133, 144, 599], 'LineWidth', 3,'Color', 'white');
%maskedImage=insertShape(maskedImage, 'Line', [619, 133, 946, 599], 'LineWidth', 3,'Color', 'white');
%imshow(maskedImage)
maskedImage = imcrop(maskedImage , boxbw);
whitearea = imcrop(whitearea , boxbw);

image_final = maskedImage;

%% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:1
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_LGDF = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);

 %% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:2
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_LSACM = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


    
        %% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:3
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_Bayes = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


         
 %% 

image = dicomread('7.dcm');
%imshow(image);


%imshow(img)
%impixelinfo
x = [471 547 620 947 547 145];
y = [133 157 133 599 724 599];

bw = poly2mask(x,y,768,1024);
%imshow(bw)



[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));

newThreshold = 0.2;


bin = imbinarize(image,newThreshold);
%imshow(bin);


maskedImage = image; % Initialize
maskedImage(~bw) = 0; % Erase background.

%imshow(maskedImage);

%new_image = ~maskedImage;
%imshow(new_image)

image = maskedImage;


[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));
% No black pixels here
newThreshold =0.1;
% finding th threshold needs to be updated

bin = imbinarize(image,newThreshold);
%imshow(bin);

binaryImage = imclearborder(bin);
%imshow(binaryImage);

%imagen = image.*bin;
%imshow(imagen);

%[B,L] = bwboundaries(bin,'holes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
%hold on
%for k = 1:length(B)
%   boundary = B{k};
%   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%end


binImage = imfill(bin,'holes');
%imshow(binImage);


binImage(~bw)=1;
binImage = ~binImage;
%imshow(binImage);
%now we extract the white area, we can select multiple ones
whitearea = bwareafilt(binImage,1);
%imshow(whitearea);


%check_point = find(whitearea);
%check_point_t = check_point(floor(length(check_point/2)));
%ismember(check_point_t,find(whitearea))
%
%test_tresh = 0.1;
%for i = 0:15
%    newThreshold = test_tresh+(i)*0.02;
%bin = imbinarize(image,'adaptive',newThreshold);
%binaryImage = imclearborder(bin);
%binImage = imfill(bin,'holes');
%binImage(~bw)=1;
%binImage = ~binImage;
%whitearea = bwareafilt(binImage,1);
%test = find(whitearea);
%res= ismember(check_point_t,test);
%if res == 0
%    final_thresh = test_tresh+(i-1)*0.02;
%    final_white_area =  whitearea;
%   break 
%end
%if i ==10
%    final_thresh = test_tresh+(i)*0.02;
%    final_white_area =  whitearea;
%end
%end



boundries = bwboundaries(whitearea);

[B,L] = bwboundaries(whitearea,'noholes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
stats = regionprops(whitearea,'area', 'Centroid','BoundingBox');
centre=stats.Centroid;
%%%%%%%%Resizing maybe?
Im = whitearea;
[rows,columns] = size(Im);
%imshow(Im)

%extract properties of the region


%%%extending using centroid

%extractin polygon containing area

boxbw = stats.BoundingBox;
boxbw(1:2) = boxbw(1:2) - 90;
boxbw(3:4) = boxbw(3:4) + 150;
%image = dicomread('4.dcm');
%imshow(maskedImage);
%maskedImage=insertShape(maskedImage, 'Line', [470, 133, 144, 599], 'LineWidth', 3,'Color', 'white');
%maskedImage=insertShape(maskedImage, 'Line', [619, 133, 946, 599], 'LineWidth', 3,'Color', 'white');
%imshow(maskedImage)
maskedImage = imcrop(maskedImage , boxbw);
whitearea = imcrop(whitearea , boxbw);

image_final = maskedImage;


%% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:50
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 70
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end

phi = imfill(phi,'holes');

%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 30; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_truth = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


  %% 
  
   figure(),
        imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
        hold on,[c,h] = contour(phi_LGDF,[0 0],'m','linewidth',3); 
        [c,h] = contour(phi_LSACM,[0 0],'g','linewidth',3);
        [c,h] = contour(phi_Bayes,[0 0],'c','linewidth',3);
        %[c,h] = contour(phi_truth,[0 0],'y','linewidth',3);
        hold off
        
  %% 
 % cd '\\iowa.uiowa.edu\shared\engineering\home\afallahdizcheh\windowsdata\Desktop\Image_segmentatation';

image = dicomread('7.dcm');
%imshow(image);


%imshow(img)
%impixelinfo
x = [471 547 620 947 547 145];
y = [133 157 133 599 724 599];

bw = poly2mask(x,y,768,1024);
%imshow(bw)



[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));

newThreshold = 0.2;


bin = imbinarize(image,newThreshold);
%imshow(bin);


maskedImage = image; % Initialize
maskedImage(~bw) = 0; % Erase background.

%imshow(maskedImage);

%new_image = ~maskedImage;
%imshow(new_image)

image = maskedImage;


[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));
% No black pixels here
newThreshold =0.1;
% finding th threshold needs to be updated

bin = imbinarize(image,newThreshold);
%imshow(bin);

binaryImage = imclearborder(bin);
%imshow(binaryImage);

%imagen = image.*bin;
%imshow(imagen);

%[B,L] = bwboundaries(bin,'holes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
%hold on
%for k = 1:length(B)
%   boundary = B{k};
%   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%end


binImage = imfill(bin,'holes');
%imshow(binImage);


binImage(~bw)=1;
binImage = ~binImage;
%imshow(binImage);
%now we extract the white area, we can select multiple ones
whitearea = bwareafilt(binImage,1);
%imshow(whitearea);


%check_point = find(whitearea);
%check_point_t = check_point(floor(length(check_point/2)));
%ismember(check_point_t,find(whitearea))
%
%test_tresh = 0.1;
%for i = 0:15
%    newThreshold = test_tresh+(i)*0.02;
%bin = imbinarize(image,'adaptive',newThreshold);
%binaryImage = imclearborder(bin);
%binImage = imfill(bin,'holes');
%binImage(~bw)=1;
%binImage = ~binImage;
%whitearea = bwareafilt(binImage,1);
%test = find(whitearea);
%res= ismember(check_point_t,test);
%if res == 0
%    final_thresh = test_tresh+(i-1)*0.02;
%    final_white_area =  whitearea;
%   break 
%end
%if i ==10
%    final_thresh = test_tresh+(i)*0.02;
%    final_white_area =  whitearea;
%end
%end



boundries = bwboundaries(whitearea);

[B,L] = bwboundaries(whitearea,'noholes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
stats = regionprops(whitearea,'area', 'Centroid','BoundingBox');
centre=stats.Centroid;
%%%%%%%%Resizing maybe?
Im = whitearea;
[rows,columns] = size(Im);
%imshow(Im)

%extract properties of the region


%%%extending using centroid

%extractin polygon containing area

boxbw = stats.BoundingBox;
boxbw(1:2) = boxbw(1:2) - 90;
boxbw(3:4) = boxbw(3:4) + 140;
%image = dicomread('4.dcm');
%imshow(maskedImage);
%maskedImage=insertShape(maskedImage, 'Line', [470, 133, 144, 599], 'LineWidth', 3,'Color', 'white');
%maskedImage=insertShape(maskedImage, 'Line', [619, 133, 946, 599], 'LineWidth', 3,'Color', 'white');
%imshow(maskedImage)
maskedImage = imcrop(maskedImage , boxbw);
whitearea = imcrop(whitearea , boxbw);

image_final = maskedImage;

%% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:3
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_LGDF2 = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


  
 %% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:5
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_LSACM2 = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


    
        %% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:10
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_Bayes2 = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


        
  %% 
  
   figure(),
        imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
        hold on,[c,h] = contour(phi_LGDF2,[0 0],'m','linewidth',3); 
        [c,h] = contour(phi_LSACM2,[0 0],'g','linewidth',3);
        [c,h] = contour(phi_Bayes2,[0 0],'c','linewidth',3);
        [c,h] = contour(phi_truth,[0 0],'y','linewidth',3);
        hold off
        
        
 %%         %% %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:7
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 65
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end


%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=maskedImage;
Img = double(Img(:,:,1));
NumIter= 10; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_Unet = phi;
%res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);


        
 
 %% proposed
ts_proposed = tic;
init_cont = whitearea*-5+2.5;
phi=init_cont;

for i = 1:50
area_cont = ones(size(maskedImage));
area_ind = find(phi<0);
area_cont(area_ind)=phi(area_ind);
  
area_cont(area_cont<0) = 0;
area_cont=~area_cont;
%imshow(area_cont);

%extend

SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(area_cont,SE);


new_area_ind = find(d-area_cont);
data_local = maskedImage(new_area_ind);
nz=data_local(data_local>0);
%imhist(nz);
thresh_low =quantile(nz,0.01);
thresh_up =quantile(nz,0.2);
if thresh_up > 67
    break
end 
check = length(new_area_ind(data_local>thresh_low & data_local<thresh_up))/length(new_area_ind);
if check <0.05
    break
end 
if mode(nz)>100
    break
end
update_area_ind = new_area_ind(data_local>thresh_low & data_local<thresh_up);

area_cont(update_area_ind) =1;
area_cont = area_cont *-5+2.5;

phi = area_cont;
end
img=maskedImage;

SE=strel('disk',30);
 
% apply the dilation operation.
d=imdilate(phi,SE);
 

SE=strel('disk',15);
 
%apply the erosion operation.
e=imerode(phi,SE);
 
%display all the images.
%imtool(k,[]);
%imtool(e,[]);
 
%see the effective reduction in org,image
%imtool(d-e,[]);
bwmask = d-e;

%nz = find(bwmask);
img(~bwmask)=0;
%imshow(img);

%clc;clear all;close all;
%image = dicomread('test.dcm');
%test_cropped = imcrop(image);
%dicomwrite(test_cropped,'test_cropped.dcm');
%image = dicomread('test_cropped.dcm');
%image=imread('new.bmp')
Img=img;
Img = double(Img(:,:,1));
NumIter= 30; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3;%size of kernel
epsilon = 1;
c0 = 1; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight
%figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);


 

%%%%%%
Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 
%figure,imagesc(uint8(Img),[0   255]),colormap(gray),axis on;axis equal,
%hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1); hold off
%pause(0.5)
%tic
   [phi lf]=evolution_proposed_initial(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);
for iter = 2:NumIter
    [phi lf]=evolution_proposed(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf,lf);
    %if(mod(iter,1) == 0)
       % figure(2),
       % imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
       % hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
       % pause(0.02);
    %end
end
phi_proposed = phi;
res_proposed=(phi_proposed<= 0);
%toc
te_proposed = toc(ts_proposed);

  figure(),
        imagesc(uint8(maskedImage),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
        hold on,[c,h] = contour(phi_proposed,[0 0],'r','linewidth',3);
       [c,h] = contour(phi_Unet,[0 0],	'color',[0.96 0.6 0.9],'linewidth',3);
        [c,h] = contour(phi_Bayes2,[0 0],'c','linewidth',3);
        [c,h] = contour(phi_truth,[0 0],'y','linewidth',3);
  
        hold off;
         
    
        