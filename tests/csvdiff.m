function csvdiff()
    %for h=1:10:41
    %for h=1:30
    for h=1:1
        %experimentalCsv_fname = ['./ffld_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        experimentalCsv_fname = ['./piggyHOG_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        %experimentalCsv_fname = ['./piotr_results/carsgraz_001.image_scale_' int2str(h) '.csv'];
        referenceCsv_fname = ['./voc5_features_results/carsgraz_001.image_scale_' int2str(h) '.csv']; %VOC5 is our baseline reference 
        mydiff(experimentalCsv_fname, referenceCsv_fname, h)
    end

%TODO: rename 'h' to 'level'
function mydiff(experimentalCsv_fname, referenceCsv_fname, h)
    referenceResult = csvread(referenceCsv_fname);
    referenceResult = unpackHog(referenceResult);
    %referenceResult = referenceResult(2:end, :); %remove header (which shows dims)

    experimentalResult = csvread(experimentalCsv_fname);
    experimentalResult = unpackHog(experimentalResult);
    %experimentalResult = experimentalResult(2:end, :); %remove header (which shows dims)

    thresh = 0.1;
    diff = abs(experimentalResult - referenceResult);

    [inHeight_voc5 inWidth_voc5] = size(referenceResult); %TODO: remove this clunky size calculation
    [inHeight inWidth] = size(experimentalResult);
    resultSize = inHeight * inWidth;

    display(['hog[' int2str(h) ']:'])
    %display(['    size(referenceResult) = ' mat2str(referenceResult(1, 1:3))]) %CSV header that shows dims
    %display(['    size(experimentalResult) = ' mat2str(experimentalResult(1, 1:3))]) 

    %display(['    nnz(diff) = ' num2str(nnz(diff))])
    display(['    percent mismatches above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)/resultSize * 100) '%'])

figure(h)
    visHog(referenceResult)
figure(h+100)
    visHog(experimentalResult)

% @param hogCsv = hog in [d w*h] that we've read from a CSV. 
% csv [d w*h] -> matlab data layout [h w d]
function hog = unpackHog(hogCsv)
    hogDims = hogCsv(1, 1:3);
    d = hogDims(1); % [d w h] = my CSV dims format in both C++ and Matlab
    w = hogDims(2);
    h = hogDims(3);

    hog_2d = hogCsv(2:end, :); %skip the [d w h] dims header
    hog_3d = reshape(hog_2d, [d w h]); % [d w*h] -> C++ style [d w h] layout
    hog = permute(hog_3d, [3 2 1]); % Matlab style [h w d] layout

function visHog(hog)
    addpath('../vis');
    w = foldHOG(hog);
    visualizeHOG(double(max(0,w)));
    %[path, imgname, ext] = fileparts(curr_img);
    %print(gcf, '-dpng', '-r0', [output_dir '/' imgname '_scale_' int2str(level) ext]);


