function FncSaveFormats(OutputFolder,filename1,res)

% The funciton saves the last figure, with the following formats:

exportgraphics(gcf, fullfile(OutputFolder, filename1 + ".jpg"), 'Resolution', res);
exportgraphics(gcf, fullfile(OutputFolder, filename1 + ".png"), 'Resolution', res);
exportgraphics(gcf, fullfile(OutputFolder, filename1 + ".tif"), 'Resolution', res);
print(fullfile(OutputFolder, filename1 + ".svg"), '-dsvg', '-r300');
print(fullfile(OutputFolder, filename1 + ".eps"), '-depsc', '-r300');

end