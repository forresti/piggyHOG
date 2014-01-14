


This is Forrest's hack to do the following...

1. Input: image filename from PASCAL VOC detection challenge
2. sample multiscale
3. Output: stich multiscale onto same-sized planes.
    (TODO: add a text file that explains where the images were placed in planes!)
    

    saves something like this:
    inputFilename_plane0.jpg
    inputFilename_plane1.jpg
    ...


#TODO: find/replace NbFeatures for NbChannels.
       set NbChannels=3

#TODO: in ffld.cpp, replace {hog, HOG, Hog} to Pyra

#TODO: in Patchwork.{cpp, h}, change the definition of 'Plane' to JPEGImage

DONE:
 find/replace NbFeatures for NbChannels.
 set NbChannels=3

