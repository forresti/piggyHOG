
%im = imread('~/vision_experiments/voc-release5/ig02-v1.0-cars/cars/carsgraz_420.image.png');
im = imread('../../images_640x480/carsgraz_001.image.png');
I=imResample(single(im)/255,[480 640]);
tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc)

