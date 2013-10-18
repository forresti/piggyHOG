
%example in typical [h w d] format
hog(1,1,:) = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
hog(1,2,:) = hog(1,1,:) + 0.1;
hog(1,3,:) = hog(1,1,:) + 0.2;

hog(2,1,:) = hog(1,1,:) + 0.3;
hog(2,2,:) = hog(1,1,:) + 0.4;
hog(2,3,:) = hog(1,1,:) + 0.5

%transposed (as in my common/writeToCsv_withSize.m)
addpath('../vis');
visualizeHOG(hog)

%transposed + squished to 2D

%save

%load

%un-squish 2D -> 3D

%un-transpose to Matlab format


%diff with input

%visualize


