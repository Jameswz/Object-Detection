function data = preprocessData(data,targetSize)
% Resize the image and scale the corresponding bounding boxes.

% Copyright 2020 The MathWorks, Inc.

[im,bboxes,labels,masks] = data{:};
imgSize = size(im);

% Resize the min dimension to targetSize and resize the other dim to
% maintain aspect ratio.

% Resize images, masks, and bboxes
[~,minDim] = min(imgSize(1:2));
resizeSize = [NaN NaN];
resizeSize(minDim) = targetSize(minDim);

im = imresize(im,resizeSize);
masks = imresize(masks,resizeSize);

resizeScale = targetSize(minDim)/imgSize(minDim);

bboxes = bboxresize(round(bboxes),resizeScale);

% Crop to target size
cropWindow = randomWindow2d(size(im),targetSize(1:2));
[bboxes,indices] = bboxcrop(bboxes,cropWindow,'OverlapThreshold',0.7);
im = imcrop(im,cropWindow);

[r,c] = deal(cropWindow.YLimits(1):cropWindow.YLimits(2),cropWindow.XLimits(1):cropWindow.XLimits(2));
masks = masks(r,c,indices);

labels = labels(indices);


if(isempty(bboxes))
    data = [];
    return;
end

bboxes = max(bboxes,1);

% Normalize image using COCO imageset means
imageMean = single([103.53, 116.28, 123.675]);

% Step 2: RGB -> BGR
im = im(:,:,[3 2 1]);

im = single(im);

% Step 3: Normalize
im(:,:,1) = im(:,:,1) - imageMean(1);
im(:,:,2) = im(:,:,2) - imageMean(2);
im(:,:,3) = im(:,:,3) - imageMean(3);


data{1} = im;
data{2} = bboxes;
data{3} = labels;
data{4} = masks;

end