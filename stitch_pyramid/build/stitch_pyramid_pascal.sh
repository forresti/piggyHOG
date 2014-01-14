#./stitch_pyramid ../../../images_640x480/carsgraz_001.image.jpg
#./stitch_pyramid --padding 8 ../../images_640x480/carsgraz_001.image.jpg

VOC_DIR=/media/big_disk/VOC2007/VOCdevkit/VOC2007
INPUT_DIR=$VOC_DIR/JPEGImages
OUTPUT_DIR=$VOC_DIR/JPEGImages_stitched_pyramid

#example...
./stitch_pyramid --padding 8 --output-stitched-dir $OUTPUT_DIR ../../images_640x480/carsgraz_001.image.jpg
#./stitch_pyramid --padding 8 --output-stitched-dir trololol ../../images_640x480/carsgraz_001.image.jpg

#for img in $INPUT_DIR/*jpg
#do

    #echo $img

#done


