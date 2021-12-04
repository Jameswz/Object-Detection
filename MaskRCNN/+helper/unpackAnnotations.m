function unpackAnnotations(trainCats,annotationFile,trainImgFolder,unpackLocation)
% Unpack COCO annotations to MAT files

% Copyright 2020 The MathWorks, Inc.

% Initialize the CocoApi object
coco = CocoApi(annotationFile);
  
% Get all image ids filtered based on train categories
catIds = coco.getCatIds('catNms',trainCats);
imgIds = coco.getImgIds('catIds',catIds);

% Create in-memory datastore to manage imageIDs
imgID_DS = arrayDatastore(imgIds);
% Get image and ground truth data from imageIds
ds = transform(imgID_DS,@(x)helper.cocoAnnotationsFromID_preprocess(x{1},coco,trainImgFolder,catIds));

disp('Unpacking annotations into MAT files...');
while (ds.hasdata)
    
    data = read(ds);
    [imageName,bbox,label,masks] = data{:};  
    imageName_Number = imageName(16:end-4);
            
    labelFilename = strcat(unpackLocation,"/label_",num2str(imageName_Number),".mat");
    save(labelFilename,'imageName','bbox','label','masks')
     
end

disp('Done!');

end