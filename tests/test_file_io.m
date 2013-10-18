
%   Experimenting with going between Matlab-style layout, C++ style layout, CSV, and back. 
%   Making sure nothing gets lost in translation

%example in typical [h w d] format
hog(1,1,:) = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
hog(1,2,:) = hog(1,1,:) + 0.1;
hog(1,3,:) = hog(1,1,:) + 0.2;

hog(2,1,:) = hog(1,1,:) + 0.3;
hog(2,2,:) = hog(1,1,:) + 0.4;
hog(2,3,:) = hog(1,1,:) + 0.5;

[h w d] = size(hog);

%visualize initial hog
addpath('../vis');
visualizeHOG(hog)

%transposed (as in my common/writeToCsv_withSize.m)
hog_transposed = permute(hog, [3 2 1]); % [h w d] -> [d w h]

%transposed + squished to 2D (as in my common/writeToCsv_withSize.m)
hog_squished_2d = reshape(hog_transposed, [d w*h]);

%save
csvwrite('temp_hog.csv', hog_squished_2d);

%load
hog_squished_2d_afterCsv = csvread('temp_hog.csv');

%un-squish 2D -> 3D
hog_transposed_afterCsv = reshape(hog_squished_2d_afterCsv, [d w h]); %[d w*h] -> [d w h]

%un-transpose to Matlab format
hog_afterCsv = permute(hog_transposed_afterCsv, [3 2 1]); % [d w h] -> [h w d]

%diff with input
diff_vs_orig = hog - hog_afterCsv;
display([' num values different before and after = ' int2str(sum(sum(sum(diff_vs_orig))))])

%visualize again
%visualizeHOG(hog_afterCsv)

