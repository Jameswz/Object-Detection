function cmap = camvid32ColorMap()
% Define the color map used by the CamVid data set for 32 classes.

cmap = [ 
    64 128 64     % Animal
    192 0 128     % Archway
    0 128 192     % Biyclist
    0 128 64      % Bridge
    128 0 0       % Building
    64 0 128      % Car
    64 0 192      % CartLuggagePram
    192 128 64    % Child
    192 192 128   % Column Pole
    64 64 128     % Fence
    128 0 192     % LaneMkgsDriv
    192 0 64      % LaneMkgsNonDriv
    128 128 64    % Misc Text 
    192 0 192     % MotorcycleScooter
    128 64 64     % OtherMoving
    64 192 128    % ParkingBlock
    64 64 0       % Pedestrian
    128 64 128    % Road
    128 128 192   % RoadShoulder
    0 0 192       % Sidewalk
    192 128 128   % SignSymbol
    128 128 128   % Sky
    64 128 192    % SUVPickupTruck
    0 0 64        % TrafficCone
    0 64 64       % TrafficLight
    192 64 128    % Train
    128 128 0     % Tree
    192 128 192   % Truck Bus 
    64 0 64       % Tunnel
    192 192 0     % VegetationMisc
    0 0 0         % Void  
    64 192 0      % Wall
 ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end
  