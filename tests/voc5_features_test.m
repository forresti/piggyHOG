
function demo()
    output_dir = 'tests/voc5_features_results'
    curr_img = 'images_640x480/carsgraz_001.image.png';

    load('VOC2007/car_final'); %load 'model' structure
    pyra = time_extract_hog(curr_img, model);

    visHog(pyra, output_dir, curr_img)

function pyra = time_extract_hog(img_name, model)
    im = imread(img_name);

    th = tic();
    pyra = featpyramid(double(im), model);
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);

% @param curr_img is just for setting an output path.
function visHog(pyra, output_dir, curr_img)
    nlevels = length(pyra.feat);

    for level = 1:10:nlevels
        figure(level)
        w = foldHOG(pyra.feat{level});
        visualizeHOG(double(max(0,w)));
        [path, imgname, ext] = fileparts(curr_img);
        print(gcf, '-dpng', '-r0', [output_dir '/scale' int2str(level) '_' imgname ext]);
    end    


