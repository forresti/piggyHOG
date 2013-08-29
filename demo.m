
function demo()
    img_dir = 'images_640x480';
    imgs = dir([img_dir '/*.png']); %640x480 car images
    %output_dir = './cascade-results-g02-v1.0-cars';
    load('VOC2007/car_final'); %load 'model' structure

    for idx = 1:length(imgs)
        curr_img = imgs(idx).name
        pyra = time_extract_hog(curr_img, model);

        visHog(pyra)
        %print(gcf, '-dpng', '-r0', [output_dir '/' curr_img]);
    end

function pyra = time_extract_hog(img_name, model)
    im = imread(img_name);
    %im = color(im);

    th = tic();
    %pyra = featpyramid(double(im), model);
    pyra = featpyramid_fhog(single(im), model);
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);

function visHog(pyra)
    %just visualize top HOG level
    
    nlevels = length(pyra.feat);

    for level = 1:10:nlevels
        figure(level)
        w = foldHOG(pyra.feat{level});
        visualizeHOG(double(max(0,w)));
    end    


