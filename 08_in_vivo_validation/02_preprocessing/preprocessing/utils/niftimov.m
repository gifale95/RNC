function varargout = niftimov(varargin)

% This function was made as an alternative to art_movie and spm_movie,
% because these functions are either very slow or cannot deal with all
% voxel size properly. In addition, the current function is able to
% create movies of brain activations overlaid on an anatomical background
% (currently only images with same dimensions, though).
%
% When working outside of the GUI, input the list of images (P). If the
% second argument is 'default', default values are used. Otherwise, all
% necessary variables need to be provided, or the GUI will open.
%
%   P: List of images from which to create a movie
%   'Movie Type':
%                 1: Raw brain images
%                 2: Movement images (default)
%                 3: Activation images (e.g. from FIR bin images)
%   'Difference Image': (only for 'Movie Type' == 2)
%                 1: Difference to first (default)
%                 2: Difference to preceding (i.e. movement derivate)
%   'Display':
%                 0: Output all frames
%                 1: Display Only (default)
%                 2: Save as Avi Only
%                 3: Display & Save as Avi
%   'Background Image': Location of image for background (only for 'Movie Type' == 3).
%   'Range': Range of image values to be displayed (necessary only for 'Movie Type' == 3).
%   'FPS': Frames per Second (used only when movie is displayed or saved) (default: 5).
%   'Save': Folder where image is being saved.

% TODO: Error 1: Contrast of images is determined automatically from images
% TODO: Error 2: indices of activation images are wrong!
% TODO: improve speed of uint8 conversion
% TODO: introduce possibility to display a selection of slices (see transform_vol.m)
% TODO: introduce possibility to display saggital or coronal (see transform_vol.m)
% TODO: make compatible with SPM2
% TODO: introduce slider as alternative to movie (see art_movie)
% TODO: find better way to overlay activations on background image
% TODO: introduce possibility to use transformation matrix for display
% TODO: reduce memory load by converting to uint8 early


% Cases of things to plot:

% a) Movie of all images --> load all images and determine display range
% automatically from the range of values within the mask (get mask)
% b) Movie of difference images --> create reference image, load all
% images, calculate difference everywhere, how to determine range?
% c) Activation images on top of the anatomy --> load anatomy and
% activation images, determine threshold between which results should be
% displayed

%% If input is provided, assign the variables accordingly
if nargin >= 1
    P = varargin{1};
    if nargin == 2
        if strcmp(varargin{2},'default')
        movie_type = 2;
        s = 1;
        movie_display = 1;
        fps = 5;
        end
    elseif nargin >2
        for i = 2:nargin-1
            switch lower(varargin{i})
                case 'movie type'
                    movie_type = varargin{i+1};
                case 'difference image'
                    s = varargin{i+1};
                case 'background image'
                    p = varargin{i+1};
                case 'range'
                    in_range = varargin{i+1};
                case 'display'
                    movie_display = varargin{i+1};
                case 'fps'
                    fps = varargin{i+1};
                case 'save'
                    loc = varargin{i+1};
            end
        end
    end
end

%% Basic settings
if nargin == 0
    P = spm_select([1 Inf],'image','Pick Image files for movie','',cd,'.img',1);
    if isempty(P), error('No images selected!'), end
end

F = spm('CreateIntWin');
% F = spm_input('!GetWin');

if ~exist('movie_type','var')
movie_type = spm_input('What type of movie should be created?',...
    '+1','m','Raw brain images|Movement images|Activation images (e.g. from FIR bin images)',[1 2 3],2);
end


switch movie_type
    case 1
        cmap = gray(256);
    case 2
        if ~exist('s','var')
        s = spm_input('Select difference image to be calculated',...
            '+2','m','Difference to first|Difference to preceding',[1 2],1);
        end
        cmap = jet(256);
    case 3
        if ~exist('p','var')
            p = spm_select(1,'image','Select background image','',cd,'.img',1);
        end
        % Determine range of values to be plotted
        if ~exist('in_range','var')
            in_range = spm_input('Enter range:','+2','r','',[1 2]);
        end
        cmap = hot(256 + 64);  cmap = cmap((1:256) + 64,:);
end

if ~exist('movie_display','var')
    movie_display = spm_input('What should be done with the movie?',...
        '+1','m','Display & Save as Avi|Save as Avi Only|Display Only|Output all frames',[3 2 1 0],1);
end

if movie_display && ~exist('fps','var')
    fps = spm_input('Enter frames per second:','+2','r',5);
end

if movie_display >=2
    if ~exist('loc','var')
        loc = spm_select(1,'dir','Select folder to store movie','',cd,'.*',1);
    end
    [l f] = fileparts(P(1,:));
    loc = fullfile(loc,[f '.avi']);
end

delete(F)
drawnow

%% Initialization

% Create reference and mask image
vol = spm_read_vols(spm_vol(P(1,:)));
ref_vol = transform_vol(vol);
% mask_vol = ref_vol>percentile(ref_vol(:),60);
mask_vol = ref_vol>0.6*mean(ref_vol(:));
% vol_mean = mean(ref_vol(mask_vol));
% vol_std = std(ref_vol(mask_vol));
% ref_vol(~mask_vol) = 0;
% ref_vol = (ref_vol-mean(ref_vol(mask_vol)))/(cutoff*std(ref_vol(mask_vol)));
sz = size(ref_vol);

if movie_type == 3
    % get background volume
    b_vol = spm_read_vols(spm_vol(p));
    b_vol = transform_vol(b_vol);
    % in the future, use resizem to reduce size of background
    % todo: check orientation
    % downsample b_vol to vol (upsampling might lead to out of memory soon)
    % then fill overlayed image with zeros around the background volume
    % how? check header files
    
end

% Preload variables
n_images = size(P,1);
% mean_image = zeros(sz);
all_images = cell(n_images,1);
all_means = zeros(n_images,1);
% all_stds = zeros(n_images,1);

%% Load images (and if necessary calculate movie images)

hdrs = spm_vol(P);

str = sprintf('Loaded image %03i/%03i\n',0,n_images);
fprintf('\n%s',str)
b = repmat('\b',1,length(str));

for i_image = 1:n_images
    
    % Load image
    fprintf([b 'Loaded image %03i/%03i\n'],i_image,n_images)
    vol = spm_read_vols(hdrs(i_image,:));
    vol = transform_vol(vol);
    
        
    if movie_type == 2
        
        if s == 1
            vol = (ref_vol-vol);
        elseif s == 2
            if i_image == 1
                previous_vol = vol;
                vol = zeros(sz); % make empty vol
            else
                curr_vol = vol; % hold vol
                vol = vol-previous_vol;
                previous_vol = curr_vol;
            end
        end
        
    end
    
    if movie_type ~= 3
        all_means(i_image) = mean(vol(mask_vol));
%         all_stds(i_image) = std(vol(mask_vol));
    end
    
    
    all_images{i_image} = vol;
    
%         mean_image = mean_image + vol;
    
end

%     mean_image = mean_image/n_images;

if movie_type == 2 && (movie_display == 1 || movie_display == 3)
    rms = zeros(n_images,1);
    for i_image = 1:n_images
        rms(i_image) = mean(all_images{i_image}(mask_vol).^2);
    end
end
    
if movie_type ~= 3 && ~exist('in_range','var')
    
    if movie_type == 1
        fact = 2; % decrease contrast of raw images by a factor
    elseif movie_type == 2
        fact = 1; % decrease contrast of difference images by a factor
    end
    
%     in_std = mean(all_stds(2:end));
    in_mean = mean(all_means(2:end));
    in_std = std(ref_vol(mask_vol)); % range determined by reference image
%     in_mean = mean(ref_vol(mask_vol)); % same with mean

    in_range = [-fact*in_std fact*in_std] + in_mean;
    
end

%% Transform to uint8

% % if movie is saved or activation images are needed, convert it
if movie_display >=2 || movie_type == 3
    if movie_type == 3 % make uint8 with values in_range
        in_range2 = in_range - in_range(1);
        for i_image = 1:n_images
            % adjust range of image to in_range
            
            % make in_range(1) the minimum
            all_images{i_image} = all_images{i_image} - in_range(1);
                        
            % cutoff exceeding values
            all_images{i_image}(all_images{i_image}>in_range2(2)) = in_range2(2);
            all_images{i_image}(all_images{i_image}<0)  = 0;
            
            % make in_range(2) to 255
            all_images{i_image} = all_images{i_image} * 255/in_range2(2);
            all_images{i_image} = uint8(all_images{i_image});
        end
    else % make uint8 with 128 as mean
        in_range2 = in_range - mean(in_range);
        for i_image = 1:n_images
            all_images{i_image} = all_images{i_image}-mean(in_range);
            all_images{i_image} = all_images{i_image} * 127/in_range2(2) + 128;
            all_images{i_image}(all_images{i_image}>255) = 255;
            all_images{i_image}(all_images{i_image}<0) = 0;
            
            all_images{i_image} = uint8(all_images{i_image});
        end
        in_range = [0 255];
    end
end

if ~movie_display % if movie is not displayed, terminate
    varargout{1} = all_images;
    return
else
    % Create movie structure
    M = struct('cdata',cell(1,n_images),'colormap',cell(1,n_images));
end

%% If overlay over background, convert images

if movie_type == 3
    
    % Normalize intensities of b_vol and convert to uint8
    if ~exist('maskvol','var')
        maskvol = b_vol>0.6*mean(b_vol(:));
    end
    b_vol_m = mean(b_vol(maskvol));
    b_vol_std = std(b_vol(maskvol));
    b_vol = (b_vol-b_vol_m)/(2.5*b_vol_std) + 0.5;
    b_vol(b_vol>1) = 1; b_vol(b_vol<0) = 0;
    b_vol = uint8(255*b_vol);
    
    % increase size of b_vol
    b_vol = repmat(b_vol,[1 1 3]);
        
    for i_image = 1:n_images
        curr_image = b_vol;
        curr_ind = all_images{i_image}>0;
        curr_image(curr_ind) = all_images{i_image}(curr_ind);
        all_images{i_image} = curr_image;
    end
    
end

%%  Make figures / movies
hfig = figure;
if movie_display == 2 % if avi only
    set(hfig,'Visible','off')
end

colormap(cmap);

for i = 1:n_images
    if movie_type == 2 && i == 1, continue, end
    h = imagesc(all_images{i},in_range);
    axis off
    set(gca,'position',[0 0 1 1],'units','normalized')
%     j = colorbar('Location','Northoutside');
%     set(j,'XTickLabel',{'','strong increase','','','','neutral','','','','strong decrease',''}')
    if movie_display == 1 || movie_display == 3
        pause(1/fps)
    end
    %     cdata = zbuffer_cdata(hfig);
    if movie_display >= 2
        M(i) = im2frame(get(h,'Cdata'),cmap);
%         orig_mode = get(hfig, 'PaperPositionMode');
%         set(hfig, 'PaperPositionMode', 'auto');
%         cdata = hardcopy(hfig, '-Dzbuffer', '-r0'); % get image automatically
%         set(hfig, 'PaperPositionMode', orig_mode);
    end
end

if movie_type == 2, M = M(2:end); end

if movie_display >= 2
    movie2avi(M,loc,'fps',fps,'compression','none');
    fprintf('Movie was saved under:\n%s\n',loc);
end

pause(1)
close(hfig)

%% If requested, plot residuals of difference to reference image
if movie_type == 2 && (movie_display == 1 || movie_display == 3)
    figure
    plot(rms)
end