
function demo_hog_visualization()
    nms_thresh = 0.3; %for neighboring detections, this is the max allowed bounding box percent overlap (for non-maximal suppression)
    close all hidden %get rid of old figures
    startup;
    load('sensys_models/car_final.mat');
    imname = 'Dir_2_Lane_3_285/1360028304-13704.jpg';

%run car detector on one image 
    im = imread(imname); % load image
    [dets, boxes, trees, detected_root_filters] = imgdetect_forTracking(im, model, model.thresh); % detect objects
    top = nms(dets, nms_thresh);
    boxes = reduceboxes(model, boxes(top,:));

    figure(1000); %img with bboxes is figure num 1000. (so that HOG figure indexing can start from 1)
    showboxes(im, boxes(:,1:4));
    components_used = dets(:, 5); %component (orientation and associated sub-model) ID

%visualize HOG filters extracted at the detection locations
    figID = 1;
    for i=top' %for each detected bounding box that survived nonmax suppression
      %visualize HOG features extracted from detected bounding box
        figure(figID)
        w = foldHOG(detected_root_filters(i).f); %convert 32-deep HOG features into a few orientation bins
        visualizeHOG(w)
    
      %visualize DPM model's internal filter that was used for this detection
        figure(figID+100) %so it's easy to find the corresponding figures
        model_root_filter = get_model_root_filter(components_used(i), model);
        w = foldHOG(model_root_filter);
        visualizeHOG(max(0, w))

        figID = figID+1;
    end
end

