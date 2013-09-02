function csvdiff()

    %for h=1:10:41
    for h=1:41
        ffldCsv = ['./ffld_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        voc5Csv = ['./voc5_features_results/carsgraz_001.image_scale_' int2str(h) '.csv'];       
        mydiff(ffldCsv, voc5Csv, h)
    end
end

function mydiff(ffldCsv, voc5Csv, h)
    ffldResult = csvread(ffldCsv);
    voc5Result = csvread(voc5Csv);
    voc5Result = voc5Result(2:end, :); %remove header (which shows dims)

    thresh = 0.1;
    diff = abs(ffldResult - voc5Result);
    [inHeight inWidth] = size(ffldResult);
    resultSize = inHeight * inWidth;
    display(['hog[' int2str(h) ']:'])
    display(['    size(result) = ' mat2str(size(ffldResult)) ' = ' num2str(resultSize)])
    %display(['    nnz(diff) = ' num2str(nnz(diff))])
    %display(['    num diff elements above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)) ])
    %display(['    percent mismatches above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)/resultSize * 100) '%'])
end

