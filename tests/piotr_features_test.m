
function piotr_features_test()

    outDir = 'tests/piotr_results';
    if ~exist(outDir)
        mkdir(outDir)
    end

    matlab_features_test('piotr');


