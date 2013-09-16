
im = imread('../../images_640x480/carsgraz_001.image.png');


nRuns = 100;
tic
for i=1:nRuns
    I=imResample(single(im)/255,[480 640]);
end
resizeFps = nRuns/toc;
disp(['resize: ' num2str(resizeFps) ' fps'])


tic 
for i=1:nRuns
     H=fhog(I,8,9); 
end
hogFps = nRuns/toc;
disp(['fhog: ' num2str(hogFps) ' fps'])

