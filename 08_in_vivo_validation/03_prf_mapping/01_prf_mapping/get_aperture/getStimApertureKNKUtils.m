close all; clear all;

try
    load stimulus_bar.mat
catch
    aperture = stimuluspreprocessing();
end

figure;
for i=1:size(aperture,3)
    imshow(aperture(:,:,i))
    title([num2str(i) '/' num2str(size(aperture,3))])
    pause(0.05)
end