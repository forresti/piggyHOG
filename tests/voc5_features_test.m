
function voc5_features_test()
    
    outDir = 'tests/voc5_features_results';
    if ~exist(outDir)
        mkdir(outDir)
    end

    matlab_features_test('voc5');


