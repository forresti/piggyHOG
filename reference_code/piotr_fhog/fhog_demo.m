
im = imread('../../images_640x480/carsgraz_001.image.png');
I=imResample(single(im)/255,[480 640]);
tic 
for i=1:100, H=fhog(I,8,9); end; 
disp([num2str(100/toc) ' fps'])

