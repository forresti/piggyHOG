function csvdiff()
    %for h=1:10:41
    for h=1:41
        experimentalCsv = ['./ffld_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        experimentalCsv = ['./piggyHOG_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        %experimentalCsv = ['./piotr_results/carsgraz_001.image_scale_' int2str(h) '.csv'];
        referenceCsv = ['./voc5_features_results/carsgraz_001.image_scale_' int2str(h) '.csv']; %VOC5 is our baseline reference 
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

    %display(['    nnz(diff) = ' num2str(nnz(diff))])
    %display(['    num diff elements above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)) ])
    display(['    percent mismatches above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)/resultSize * 100) '%'])
end

%if one of the HOG results is bigger than the other, trim it down.
% naive: just taking the top-left corner when trimming.
%[referenceResult experimentalResult] = trimToMinSize(referenceResult, experimentalResult)
    %[h w d] = reshape.... %ah, crap, need the original dims.
    %minWidth = 

