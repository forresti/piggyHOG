
%adjusted slightly for our Caffe idea

sbin = 4; 
interval = 10;
sc = 2^(1/interval);
imsize = [480 640];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));

totalPixels = 0; %sum over all scales' (width*height)

%for i = 1:interval
for i = 1:max_scale
    currScale = 1/sc^(i-1);
    currX = round(imsize(1)*currScale);
    currY = round(imsize(2)*currScale);

    currNumPixels = currX * currY
    totalPixels = totalPixels + currNumPixels;    
end

display(['totalPixels = ' num2str(totalPixels) ' sqrt(totalPixels) = ' num2str(sqrt(totalPixels))])

