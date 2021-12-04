%RoadAndSkySegmentation Automation algorithm for pixel labeling.
%   RoadAndSkySegmentation is an automation algorithm for labeling pixels
%   in a scene using a semantic segmentation network trained on 11 classes
%   using the CamVid dataset. These classes include "Sky", "Building",
%   "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car",
%   "Pedestrian" and "Bicyclist". This example only uses "Road" and "Sky".
%
%   See also groundTruthLabeler, imageLabeler, 
%   vision.labeler.AutomationAlgorithm.

% Copyright 2017 The MathWorks, Inc.

classdef SemanticSegmentation < vision.labeler.AutomationAlgorithm
    
    %----------------------------------------------------------------------
    % Algorithm Description
    properties(Constant)

        %Name
        %   Character vector specifying name of algorithm.
        Name = 'SemanticSegmentation'
        
        %Description
        %   Character vector specifying short description of algorithm.
        Description = 'This algorithm uses semanticseg with a pretrained network'
        
        %UserDirections
        %   Cell array of character vectors specifying directions for
        %   algorithm users to follow in order to use algorithm.
        UserDirections = {...
            ['Automation algorithms are a way to automate manual labeling ' ...
            'tasks. This AutomationAlgorithm automatically creates pixel ', ...
            ], ...
            ['Review and Modify: Review automated labels over the interval ', ...
            'using playback controls. Modify/delete/add ROIs that were not ' ...
            'satisfactorily automated at this stage. If the results are ' ...
            'satisfactory, click Accept to accept the automated labels.'], ...
            ['Accept/Cancel: If results of automation are satisfactory, ' ...
            'click Accept to accept all automated labels and return to ' ...
            'manual labeling. If results of automation are not ' ...
            'satisfactory, click Cancel to return to manual labeling ' ...
            'without saving automated labels.']};
    end
    
    %---------------------------------------------------------------------
    % Properties
    properties
        % Network saves the SeriesNetwork object that does the semantic
        % segmentation.
        PretrainedNetwork
        
        % Categories holds the default 'background' categorical types.
        AllCategories = {'background'};
        
        % Store names.
        Sky
        Building
        Pole
        Road
        Pavement
        Tree
        SignSymbol
        Fence
        Car
        Pedestrian
        Bicyclist
        
    end
    
    %----------------------------------------------------------------------
    % Setup
    methods
        
        function isValid = checkLabelDefinition(algObj, labelDef)
            % Allow any labels that are of type 'PixelLabel', and are named
            isValid = false;
            if (strcmpi(labelDef.Name, 'Sky') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Sky = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Building') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Building = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Pole') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Pole = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Road') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Road = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Pavement') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Pavement = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Tree') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Tree = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'SignSymbol') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.SignSymbol = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Fence') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Fence = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Car') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Car = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Pedestrian') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Pedestrian = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
             elseif (strcmpi(labelDef.Name, 'Bicyclist') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.Bicyclist = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
            elseif(labelDef.Type == labelType.PixelLabel)
                % Only labels for PixelLabel ROI's are considered valid.
                isValid = true;
            end
        end
    end
    
    %----------------------------------------------------------------------
    % Execution
    methods
        
        function initialize(algObj, ~)
            
            % Point to tempdir where pretrainedSegNet was downloaded.
            pretrainedFolder = fullfile('.','pretrainedNetwork');
            pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat'); 
            data = load(pretrainedSegNet);
            % Store the network in the 'Network' property of this object.
            algObj.PretrainedNetwork = data.net;
        end
        
        function autoLabels = run(algObj, I)
            
            % Setup categorical matrix with categories.
            autoLabels = categorical(zeros(size(I,1), size(I,2)),0:11,algObj.AllCategories,'Ordinal',true);
            
            pixelCat = semanticseg(I, algObj.PretrainedNetwork);
            if ~isempty(pixelCat)
                % Add the selected pixel label at position(s)
               autoLabels(pixelCat == 'Sky') = algObj.Sky;
               autoLabels(pixelCat == 'Building') = algObj.Building;
               autoLabels(pixelCat == 'Pole') = algObj.Pole;
               autoLabels(pixelCat == 'Road') = algObj.Road;
               autoLabels(pixelCat == 'Pavement') = algObj.Pavement;
               autoLabels(pixelCat == 'Tree') = algObj.Tree;
               autoLabels(pixelCat == 'SignSymbol') = algObj.SignSymbol;
               autoLabels(pixelCat == 'Fence') = algObj.Fence;
               autoLabels(pixelCat == 'Car') = algObj.Car;
               autoLabels(pixelCat == 'Pedestrian') = algObj.Pedestrian;
               autoLabels(pixelCat == 'Bicyclist') = algObj.Bicyclist;
            end    
        end
    end
end