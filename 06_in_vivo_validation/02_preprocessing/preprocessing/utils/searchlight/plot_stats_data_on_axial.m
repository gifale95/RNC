%% creates a figure that shows data on top of axial slices of a MNI T1 brain
% navigate to the directory that has the ROIs in it
% this script assumes that you are running it from the directory where the m file is

function pic = plot_stats_data_on_axial(data_path, data2_path,fig_title)

% set planes to show 
show_planes=[26:4:52] % axial the planes throught the T1 to be shown

% load background T1
hdr_bg=spm_vol('/afs/cbs.mpg.de/software/spm/12.7771/9.10/bionic/canonical/single_subj_T1.nii'); % load the backgroud T1 volume; this is usually in yr directory of spm installation /canonical/
T1=spm_read_vols(hdr_bg); %T1

% load overlay example
hdr_overlay=spm_vol(data_path); % load the overlay, this is some example form me. It is NaN where nothing is to be displayed, and 0 to some other value elsewhere
overlay=spm_read_vols(hdr_overlay); %overlay volume
sz_overlay=size(overlay); % we need to know the size of the overlay volume for further processing
if nargin>1
    
    hdr_overlay2 = spm_vol(data2_path); 
    overlay2 = spm_read_vols(hdr_overlay2);
    
    %mask2 = spm_read_vols(spm_vol(mask2_path)); 
end 

% set a color scale between 0 and the maximum in the overlay for displaying the overlay (this is just a choice we make, you can of course make a
% different choice
if nargin > 1
tmp=overlay(:)+overlay2(:).*-1; 
overlap = intersect(find(overlay==1), find(overlay2==1)); 
tmp(overlap)=0.5;
min_val=-1; % set the minimal value for display to -1 if two overlays are displayed
overlay = overlay+overlay2*-1;
overlay(overlap) = 0.5;
else 
    tmp = overlay(:);
    min_val = 0; %otherwise set min val to 0 
end 
max_val=nanmax(tmp); %find min of the overlay
color_limits= linspace(min_val,max_val,100) %scaled globally to overall min max over time
colors=jet(numel(color_limits)*2); % that is a workaround to make the colors actually the jet colormap from half of its values...
colors=colors(numel(color_limits)+1:end,:); % so we end up with a color value for each of the 100 steps    

% Creating T1 image with overlay in it. The idea is to put the overlay (which is in color) into the T1 (which is in
% grayscale)
overlayed_T1=repmat(T1,1,1,1,3); %start with the overlay T1 being equal to the T1 in 3 dimensions (it wa 1 before b?c only black and white, we need 3 dimensions for color
sz_overlayed_T1=size(overlayed_T1); % we need to know the size of the overlayed T1 for further processing

%clear get_out_final get_out cord_overlay_space_dump
[x y z]=meshgrid(1:sz_overlayed_T1(1),1:sz_overlayed_T1(2),1:sz_overlayed_T1(3)); % create a meshgrid over the size of the T1
zz=[x(:),y(:),z(:),ones(numel(x),1)]; % variable of x,y,z linearized and 1s (for 4D matrix transformation

cord_overlay_space_dump= hdr_overlay.mat \ (hdr_bg.mat * zz'); %transformation matrix from bg to overlay, so the coordinates in T1 are set to overlay space
cord_overlay_space_dump=round(cord_overlay_space_dump); % round, as indices are full numebrs

%eliminate coords that are outside of the overlay
% get out is a funny stucture, it has rows (voxels) * columns (criteria). to exclude voxels that do not fit the criteria
get_out(1,:)=mean(cord_overlay_space_dump<1,1)>0; %index to columns that have numbers smaller than 0
get_out(2,:)=cord_overlay_space_dump(1,:)>sz_overlay(1);
get_out(3,:)=cord_overlay_space_dump(2,:)>sz_overlay(2);
get_out(4,:)=cord_overlay_space_dump(3,:)>sz_overlay(3);

get_out_final=mean(get_out,1)>0; %only those indices that fulfill all criteria

cord_overlay_space_dump(:,get_out_final)=[]; % remove impossibel coords
zz_dump=zz; zz_dump(get_out_final,:)=[]; % a new variable of meshgrid coordinates to voxels that exist

for i=1:size(zz_dump,1) %all overlay coords that fit into the anatomical
    cord_overlay_space= cord_overlay_space_dump(1:3,i); %cords in overlay space
    if overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3))>= min_val && ...
            overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3)) ~= 0 % this is my criterion- is the value there >0, You could have a very different criterion
        color=overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3)); % a specific value in the overlay
        color_code=find(abs(color_limits-color)==min(abs(color_limits-color)),1); % on a scale of 1-100, where is that wrt the min and max
        overlayed_T1(zz_dump(i,1),zz_dump(i,2),zz_dump(i,3),:) = colors(color_code,:) ; %fill up the T1 at the location of that voxel with the color looked up
    end
end

% plot the T1 with the overlay on top
fh=figure('Position',[100,500,800,300],'Color',[0 0 0]');
set(gcf, 'units','normalized','outerposition',[0 0 0.85 0.5]);
colormap(colors);
for i=1:numel(show_planes)
    sa(i)=subaxis(1,numel(show_planes),i,'SpacingVert',0.04 ,'SpacingHoriz', 0,'MR',0,'ML',0,'PT',0,'PB',0);
    fixed_image=squeeze(overlayed_T1(:,:,show_planes(i),:,:)); %axial slice
    imagesc(rot90(fixed_image));
    axis equal tight off
    
    %calculate axial slice position in real world
    tmp_mat=hdr_overlay.mat*[1 1 show_planes(i) 1]';
    axial_slice_nr=tmp_mat(3);
    if i == ceil(numel(show_planes)/2)
        tl(i)=title(fig_title,'FontSize',16,'VerticalAlignment','baseline','Color',[1 1 1]);
    end 
end

% add a colorbar
colormap(colors)
set(gca,'CLim',[min_val max_val])


set(sa(numel(sa)),'CLim',[min_val max_val])
cl=colorbar('southoutside')
set(cl,'Position',[0.8 0.20 0.15 0.03])
orig_ticks=get(cl,'Ticks')
new_ticks=[min_val max_val];
set(cl,'Ticks',new_ticks,'FontSize',14,'Color',[1 1 1])
%xl =xlabel(cl,['Spearmans''s R']);
%set(xl,'FontSize',16)
%set(xl,'Position',[1.5 0.0709 0],'FontSize',16)
%cl.Label.String='R'

pic=getframe(gcf); %gat a frame of current figure
end 