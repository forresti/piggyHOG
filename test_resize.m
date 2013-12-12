
img = imread('./images_640x480/carsgraz_001.image.jpg');

%voc5 bilinear interp
img_resize_voc5 = resize(double(img), 0.7); %bilinear interp
figure(1)
imshow(uint8(img_resize_voc5))


%matlab bilinear interp
img_resize_bilinear_builtin = imresize(img, 0.7, 'bilinear');
figure(2)
imshow(img_resize_bilinear_builtin)

%matlab nearest-neighbor interp
img_resize_nearest_builtin = imresize(img, 0.7, 'nearest');
figure(3)
imshow(img_resize_nearest_builtin)


