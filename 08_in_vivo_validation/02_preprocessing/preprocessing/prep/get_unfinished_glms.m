derived_dir = '/scratch/singej96/rcor_collab/derived';

for sub =1 :31
    fprintf('sub%02i\n',sub)
 sub_dir =fullfile(derived_dir,sprintf('sub%02i',sub),'results', 'GLM','hrf_fitting');
 sub_folders = {dir(sub_dir).name}';
 % Define a custom function to check if a string represents a number between 1 and 20
isStringNumberBetween1and20 = @(x) ~isnan(str2double(x)) && str2double(x) >= 1 && str2double(x) <= 20;

% Use cellfun to apply the custom function to each cell in the cell array
logicalIndices = cellfun(isStringNumberBetween1and20, sub_folders);

% Find the actual strings that represent numbers between 1 and 20
stringsBetween1and20 = str2double(sub_folders(logicalIndices));

sub_max(sub)=max(stringsBetween1and20);

for i =1:sub_max(sub)
        fprintf('HRF %02i\n',i)
        
    if exist(fullfile(derived_dir,sprintf('sub%02i',sub),'results', 'GLM','hrf_fitting',num2str(i),'Res_7520.nii')), sub_max(sub)=i;end
end
end

disp(sub_max)