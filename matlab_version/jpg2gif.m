function jpg2gif(delay, positionfig, position, ext)

figPatch = strcat('.\', position, '\');
figName = dir(strcat(figPatch, '*', ext));
figNum = length(figName);

% fig1 = figure('Position',positionfig);
fig1 = figure();
% set(fig1, 'visible', 'off');
for i0 = 1:figNum
  temp1 = imread([figPatch, figName(i0).name]);
  image(temp1);
  drawnow;
  im = frame2im(getframe(fig1));
  [temp, map] = rgb2ind(im, 256);
  if i0 == 1;
    imwrite(temp, map, [figPatch, 'movie.gif'],...
      'LoopCount', Inf,...
      'DelayTime', delay);
  else
    imwrite(temp, map, [figPatch, 'movie.gif'],...
      'WriteMode', 'append',...
      'DelayTime', delay);
  end
end;

end