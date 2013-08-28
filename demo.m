
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
    pyra = featpyramid(double(im), model);
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);

function visHog(pyra)
    %just visualize top HOG level
    
    w = foldHOG(pyra.feat{1});
    visualizeHOG(double(max(0,w)));
    

