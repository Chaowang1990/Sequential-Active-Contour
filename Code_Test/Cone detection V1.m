
clear

%cd 'C:\Users\afallahdizcheh\Desktop\Images';
cd '\\iowa.uiowa.edu\shared\engineering\home\afallahdizcheh\windowsdata\Desktop\Image_segmentatation';

image = dicomread('4.dcm');
imshow(image);
impixelinfo
my_vertices = [471 133;547 157;620 133;947 599;547 724;145 599];
h = drawpolygon('Position',my_vertices);

x = [471 547 620 947 547 145];
y = [133 157 133 599 724 599];

bw = poly2mask(x,y,768,1024);
imshow(bw)
hold on
plot(x,y,'b','LineWidth',2)
hold off


[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));

newThreshold = 0.2;


bin = imbinarize(image,newThreshold);
imshow(bin);


maskedImage = image; % Initialize
maskedImage(~bw) = 0; % Erase background.

imshow(maskedImage);

image = maskedImage;


[ii,jj] = find(bw);
coneintensity = image(sub2ind(size(image),ii,jj));
% No black pixels here
newThreshold = 0.12;
% finding th threshold needs to be updated

bin = imbinarize(image,newThreshold);
imshow(bin);

binaryImage = imclearborder(bin);
imshow(binaryImage);

imagen = image.*bin;
imshow(imagen);

%[B,L] = bwboundaries(bin,'holes');
%imshow(label2rgb(L, @jet, [.5 .5 .5]))
%hold on
%for k = 1:length(B)
%   boundary = B{k};
%   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
%end


binImage = imfill(bin,'holes');
imshow(binImage);

binImage(~bw)=1;
binImage = ~binImage
imshow(binImage);

%now we extract the white area, we can select multiple ones
whitearea = bwareafilt(binImage,1);
imshow(whitearea);

boundries = bwboundaries(whitearea);

[B,L] = bwboundaries(whitearea,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
%extract properties of the region

stats = regionprops(whitearea,'area', 'Centroid','BoundingBox');

%extractin polygon containing area

boxbw = stats.BoundingBox;

%image = dicomread('4.dcm');
imshow(maskedImage);
%impixelinfo
hold on;
rectangle('Position',boxbw)

xpoly = [boxbw(1) boxbw(1) boxbw(1)+boxbw(3) boxbw(1)+boxbw(3) boxbw(1) ];
ypoly = [boxbw(2) boxbw(2)+boxbw(4) boxbw(2)+boxbw(4) boxbw(2) boxbw(2)];

bwmask = poly2mask(xpoly,ypoly,768,1024);
imshow(bwmask)

%tic
%cont = activecontour(maskedImage,bwmask,2000);
%imshow(cont);
%toc

%figure
%imshow(labeloverlay(maskedImage,cont));

%tic
%cont2 = activecontour(maskedImage,bw,2000);
%imshow(cont2);
%toc
%figure
%imshow(labeloverlay(maskedImage,cont2));

% let's try to extract what we are looking for!

cur = B{1};
x=cur(:,1);
y=cur(:,2);
A=0.8;
xt_in = A*x+(1-A)*mean(x(1:end-1));
yt_in = A*y+(1-A)*mean(y(1:end-1));

F=1.2;
xt_o = F*xt_in+(1-F)*mean(xt_in(1:end-1));
yt_o = F*yt_in+(1-F)*mean(yt_in(1:end-1));

ver_in=cat(2,yt_in,xt_in);
ver_out=cat(2,yt_o,xt_o);

%imshow(maskedImage)
%h = drawpolygon('Position',ver_in);
%ho = drawpolygon('Position',ver_out);
flip_y = flip(yt_o);
flip_x = flip(xt_o);
y_over = cat(1,yt_in,flip_y);
x_over = cat(1,xt_in,flip_x);

imshow(maskedImage)
ver = cat(2,y_over,x_over);
%h = drawpolygon('Position',ver_in);
ho = drawpolygon('Position',ver);

bw = poly2mask(y_over,x_over,768,1024);
imshow(bw)
hold on
plot(y_over,x_over,'b','LineWidth',2)
hold off


tic
cont2 = activecontour(maskedImage,bw,1000);
imshow(cont2);
toc
figure
imshow(labeloverlay(maskedImage,cont2));