function csvdiff()

    %resolution='640x480';
    %resolution='1600x1200';
    resolution='2560x1600';

    %mydiff('toyConvolved_sse.csv', 'toyConvolved_ref.csv', 0, 0)

    for h=1:30
        for f=1:54
            forrestCsv = ['./output/output_hog_' int2str(h) '_filter_' int2str(f) '_000034.csv'];
            referenceCsv = ['./matlab_output/output_hog_' int2str(h) '_filter_' int2str(f) '_000034.csv'];
            %forrestCsv = ['./' resolution '_output/output_hog_' int2str(h) '_filter_' int2str(f) '_000034.csv'];
            %referenceCsv = ['./' resolution '_matlab_output/output_hog_' int2str(h) '_filter_' int2str(f) '_000034.csv'];
            mydiff(forrestCsv, referenceCsv, h, f)
        end
    end
end

function mydiff(forrestCsv, referenceCsv, h, f)
            forrestResult = csvread(forrestCsv);
            referenceResult = csvread(referenceCsv);
            referenceResult = referenceResult(2:end, :); %remove header (which shows dims)

            thresh = 0.0001;
            diff = abs(forrestResult - referenceResult);
            resultSize = size(forrestResult);
            display(['hog[' int2str(h) ']*filter[' int2str(f) ']:'])
            display(['    size(result) = ' mat2str(size(forrestResult)) ' = ' num2str(resultSize(1)*resultSize(2))])
            display(['    nnz(diff) = ' num2str(nnz(diff))])
            display(['    num diff elements above ' num2str(thresh) ' = ' num2str(nnz(diff>=thresh)) ])
end

