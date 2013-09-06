%tests for Matlab/Mex-based HOG extractors.

%@param hogMethod = 'voc5' or 'piotr'
function matlab_features_test(hogMethod)

    if(strcmp(hogMethod, 'voc5') == 1)
        output_dir = 'tests/voc5_features_results';
    elseif (strcmp(hogMethod, 'piotr') == 1) 
        output_dir = 'tests/piotr_results';
    else
        error('unknown HOG extraction method. use hogMethod={voc5 or piotr}')
    end

    curr_img = 'images_640x480/carsgraz_001.image.png';

    load('VOC2007/car_final'); %load 'model' structure
    pyra = time_extract_hog(curr_img, model, hogMethod);

    printHogSizes(pyra)
    display('writing HOG features to CSV files...');

    transpose_and_writeCsv(pyra, output_dir, curr_img)
    %visHog(pyra, output_dir, curr_img)

function pyra = time_extract_hog(img_name, model, hogMethod)
    im = imread(img_name);

    th = tic();

    if(strcmp(hogMethod, 'voc5') == 1)
        pyra = featpyramid(double(im), model);
    elseif(strcmp(hogMethod, 'piotr') == 1)
        %TODO: model.padx--;
        %      model.pady--;
        pyra = featpyramid_fhog(single(im), model);
    end
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);

function transpose_and_writeCsv(pyra, output_dir, curr_img)
    nlevels = length(pyra.feat);
    pyra = transposePyra(pyra);

    %for level = 1:10:nlevels
    for level = 1:nlevels
        [path, imgname, ext] = fileparts(curr_img);
        %csvwrite([output_dir '/' imgname '_scale_' int2str(level) '.csv'], pyra.feat{level});
        writeToCsv_withSize([output_dir '/' imgname '_scale_' int2str(level) '.csv'], pyra.feat{level});
    end

% @input dims:  (y, x, d)
% @output dims: (d, x, y) -- same as FFLD.
%                d + x*depth + y*depth*width
function pyra = transposePyra(pyra)
    nlevels = length(pyra.feat);
    for level = 1:nlevels
        pyra.feat{level} = permute(pyra.feat{level}, [3 2 1]); %[y x d] -> [d x y]
    end

% @param curr_img is just for setting an output path.
% visHog does NOT want transposed HOG features... it wants (y, x, d).
function visHog(pyra, output_dir, curr_img)
    nlevels = length(pyra.feat);

    for level = 1:10:nlevels
        figure(level)
        w = foldHOG(pyra.feat{level});
        visualizeHOG(double(max(0,w)));
        [path, imgname, ext] = fileparts(curr_img);
        print(gcf, '-dpng', '-r0', [output_dir '/' imgname '_scale_' int2str(level) ext]);
    end

%for best results, call this BEFORE transposing the pyramid.
function printHogSizes(pyra)
    nlevels = length(pyra.feat);
    for level = 1:nlevels
        [h w d] = size(pyra.feat{level});
        display(sprintf('level %d: width=%d, height=%d, depth=%d', level, w, h, d))
    end





