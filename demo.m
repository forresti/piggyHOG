
function demo()
    img_dir = 'ig02-v1.0-cars/cars';
    imgs = dir([car_dir '/*.png']) %640x480 car images
    %output_dir = './cascade-results-g02-v1.0-cars';
    load('VOC2007/car_final');

    for idx = 1:length(car_imgs)
        car_img = car_imgs(idx).name
        test([car_dir '/' car_img], model, useCascade);
        print(gcf, '-dpng', '-r0', [output_dir '/' car_img]);
    end


function pyra = time_extract_hog(im)
    th = tic();
    pyra = featpyramid(double(im), csc_model);
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);


