

function calculate_total_scales()
    bool_enforce_padding = true;
    calculate_total_scales_simple(bool_enforce_padding)

end

% adjusted slightly for our Caffe idea
% simplified (not doing the 'recycle downsampled image with various strides' thing)
% @param bool_enforce_padding = whether to add padding AFTER downsampling (to have some space between images and avoid polluting receptive fields)
function calculate_total_scales_simple(bool_enforce_padding)
    %adjusted slightly for our Caffe idea
    sbin = 4; 
    interval = 10;
    sc = 2^(1/interval);
    %imsize = [480 640];
    %imsize = [500 500];
    imsize = [2000 2000];
    max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));

    inputNumPixels = imsize(1)*imsize(2); %width*height for input img
    totalPixels = 0; %sum over all scales' (width*height)

    %for i = 1:interval
    for i = 1:max_scale
        currScale = 1/sc^(i-1);
        if(bool_enforce_padding)
            %add some padding so we don't pollute receptive fields
            currX = round(imsize(1)*currScale) + 16;
            currY = round(imsize(2)*currScale) + 16;
        else
            currX = round(imsize(1)*currScale);
            currY = round(imsize(2)*currScale);
        end

        currNumPixels = currX * currY
        totalPixels = totalPixels + currNumPixels;    
    end


    display(['    number of scales = ' num2str(max_scale)])
    display(['    pyramid: totalPixels = ' num2str(totalPixels) ' sqrt(totalPixels) = ' num2str(sqrt(totalPixels))])
    display(['    input image: inputNumPixels = ' num2str(inputNumPixels) ' sqrt(inputNumPixels) = ' num2str(sqrt(inputNumPixels))])

    display(['    ratio of pyramid vs. input img: ' num2str(totalPixels/inputNumPixels)])
end

