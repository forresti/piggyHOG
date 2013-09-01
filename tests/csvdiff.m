function csvdiff()

    for h=1:10:41
        ffldCsv = ['./ffld_results/level' int2str(h-1) '.csv']; %h-1 for C++ 0-indexing
        voc5Csv = ['./voc5_features_results/carsgraz_001.image_scale_' int2str(h) '.csv'];       
        mydiff(ffldCsv, voc5Csv, h)
    end
end

function mydiff(ffldCsv, voc5Csv, h)
            ffldResult = csvread(ffldCsv);
            voc5Result = csvread(voc5Csv);
            voc5Result = voc5Result(2:end, :); %remove header (which shows dims)

keyboard

            thresh = 0.0001;
            diff = abs(ffldResult - voc5Result);
            resultSize = size(ffldResult);
            display(['hog[' int2str(h) ']*filter[' int2str(f) ']:'])
            display(['    size(result) = ' mat2str(size(ffldResult)) ' = ' num2str(resultSize(1)*resultSize(2))])
            display(['    nnz(diff) = ' num2str(nnz(diff))])
            display(['    num diff elements above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)) ])
end

