% This code is supplementary material for the project of the course
% "Clustering algorithms"

clear
format compact
close all

load Salinas_Data

[p,n,l]=size(Salinas_Image); % Size of the Salinas cube

%Depicting the bands of the Salinas cube
 for i=1:l
     figure(1); imagesc(Salinas_Image(:,:,i))
     %pause(0.1)
 end

% Making a two dimensional array whose rows correspond to the pixels and
% the columns to the bands, containing only the pixels with nonzero label.
X_total=reshape(Salinas_Image, p*n,l);
L=reshape(Salinas_Labels,p*n,1);
existed_L=(L>0);   %This contains 1 in the positions corresponding to pixels with known class label
X=X_total(existed_L,:);
[px,nx]=size(X); % px= no. of rows (pixels) and nx=no. of columns (bands)

%...---> cl_label


% The following code can be used after the execution of an algorithm 
% Let "cl_label" be the px-dimensional vector, whose i-th element is the label
% of the class where the i-th vector in X has been assigned
% The code below, helps in depicting the results as an image again

%cl_label_tot=zeros(p*n,1);
%cl_label_tot(existed_L)=cl_label;
%im_cl_label=reshape(cl_label_tot,p,n);
%figure(10), imagesc(im_cl_label), axis equal
%figure(11), imagesc(Salinas_Labels), axis equal