
function demo()
    


function pyra = time_extract_hog(im)
    th = tic();
    pyra = featpyramid(double(im), csc_model);
    tF = toc(th);
    fprintf('  --> HOG pyramid extraction took %f seconds\n', tF);


