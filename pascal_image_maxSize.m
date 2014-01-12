
function pascal_image_maxSize()
    pascal_dir = '/media/big_disk/datasets/VOC2007/JPEGImages'

    images = dir(pascal_dir);
    maxHeight = 0;
    maxWidth = 0;

    for imgIdx = 1:length(images)
        imgFname = [pascal_dir '/' images(imgIdx).name];

        try
            im = imread(imgFname);
            [h w d] = size(im);         

            if(h > maxHeight)
                maxHeight = h
            end
            if(w > maxWidth)
                maxWidth = w
            end
        catch %'.' and '..' are in 'images' array
        end

    end

    display(['maxHeight = ' int2str(maxHeight)])
    display(['maxWidth = ' int2str(maxWidth)])
end

