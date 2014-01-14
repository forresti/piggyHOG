


This is Forrest's hack to do the following...

1. Input: image filename from PASCAL VOC detection challenge
2. sample multiscale
3. Output: stich multiscale onto same-sized planes.
    (TODO: add a text file that explains where the images were placed in planes!)
    

    saves something like this:
    inputFilename_plane0.jpg
    inputFilename_plane1.jpg
    ...

#TODO: in ffld.cpp, call Patchwork(), with MaxRows_ and MaxCols_ as 'biggest pyra scale, rounded up to a factor of 16'

#TODO: padding...
       add a JPEGImage::pad() function that creates a padded copy.

#TODO: in Patchwork.{cpp, h}, change the definition of 'Plane' to JPEGImage

#TODO: in ffld.cpp, replace {hog, HOG, Hog} to Pyra

DONE:
 find/replace NbFeatures for NbChannels.
 set NbChannels=3

