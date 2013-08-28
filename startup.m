% Prepares your matlab workspace for using voc-release5 / piggyHOG.
global G_STARTUP;

if isempty(G_STARTUP)
  G_STARTUP = true;

  % Avoiding addpath(genpath('.')) because .git includes
  % a VERY large number of subdirectories, which makes 
  % startup slow

  incl = {'images_640x480', 'reference_code', 'vis'};
  for i = 1:length(incl)
    addpath(genpath(incl{i}));
  end
end