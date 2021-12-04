function  out = cocoAnnotationMATReader(filename,trainImgFolder)

% Copyright 2020 The MathWorks, Inc.

load(filename,"imageName","bbox","label","masks");

im = imread(strcat(trainImgFolder,filesep,imageName));
% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels
if(size(im,3)==1)
    im = repmat(im, [1 1 3]);
end

out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end