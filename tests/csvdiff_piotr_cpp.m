function csvdiff()
    %for h=1:10:41
    %for h=1:41
    for h=1:1
        %experimentalCsv = ['./ffld_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing

        experimentalCsv = ['piotr_fhog_cpp_results/level0.csv']; %just doing 1 level at the moment ... corresponds to 'level 11' in pyramid 
        referenceCsv = ['./piotr_results/carsgraz_001.image_scale_' int2str(h) '.csv'];
        mydiff(experimentalCsv, referenceCsv, h)
    end
end

function mydiff(experimentalCsv, referenceCsv, h)
    referenceResult = csvread(referenceCsv);
    referenceResult = referenceResult(2:end, :); %remove header (which shows dims)

    experimentalResult = csvread(experimentalCsv);
    %experimentalResult = experimentalResult(2:end, :); %remove header (which shows dims)

    thresh = 0.1;
    diff = abs(experimentalResult - referenceResult);
    [inHeight_voc5 inWidth_voc5] = size(referenceResult);
    [inHeight inWidth] = size(experimentalResult);

    resultSize = inHeight * inWidth;
    display(['hog[' int2str(h) ']:'])
    %display(['    size(referenceResult) = ' mat2str(referenceResult(1, 1:3))]) %CSV header that shows dims
    %display(['    size(experimentalResult) = ' mat2str(experimentalResult(1, 1:3))]) 

    %display(['    size(referenceResult) = ' mat2str(size(referenceResult)) ' = ' num2str(inHeight_voc5*inWidth_voc5)])
    %display(['    size(experimentalResult) = ' mat2str(size(experimentalResult)) ' = ' num2str(resultSize)])

    display(['    nnz(diff) = ' num2str(nnz(diff))])
    display(['    percent mismatches above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)/resultSize * 100) '%'])

    ref_sum = visualizeHogSum(referenceResult);
    exp_sum = visualizeHogSum(experimentalResult);
keyboard

end

%@input one HOG, e.g. referenceResult
%TODO: input h, w, d ... or, unflatten HOG to h,w,d size first
function hog_sum = visualizeHogSum(hog)
    %temporary width, height, depth hard-coded
    h = 60;
    w = 80;
    d = 32;

    hog_3 = reshape(hog, [w h d]);
    hog_sum = squeeze(sum(hog_3, 3)); %now, size = [h w]

    HeatMap(hog_sum)
end


