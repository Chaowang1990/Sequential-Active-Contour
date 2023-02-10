%morphology


k=whitearea;
SE=strel('disk',10);
 
% apply the dilation operation.
d=imdilate(k,SE);
 
%display all the images.
%imtool(k,[]);
%imtool(d,[]);
 
%see the effective expansion
% in orginal image
%imtool(d-k,[]);


SE=strel('disk',10);
 
%apply the erosion operation.
e=imerode(k,SE);
 
%display all the images.
%imtool(k,[]);
%imtool(e,[]);
 
%see the effective reduction in org,image
imtool(d-e,[]);


k=whitearea;
SE=strel('disk',5);
 
% apply the dilation operation.
d=imdilate(k,SE);
 
%display all the images.
%imtool(k,[]);
%imtool(d,[]);
 
%see the effective expansion
% in orginal image
%imtool(d-k,[]);


SE=strel('disk',5);
 
%apply the erosion operation.
e=imerode(k,SE);
 
%display all the images.
%imtool(k,[]);
%imtool(e,[]);
 
%see the effective reduction in org,image
imtool(d-e,[]);
bwmask = d-e;