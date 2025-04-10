%% creates a figure that shows data on top of axial slices of a MNI T1 brain
% navigate to the directory that has the ROIs in it
% this script assumes that you are running it from the directory where the m file is

function [pic,fh] = plot_roi_on_struct(struct_path,mask_hdr, evc_mask, loc_mask,ppa_mask,fig_title,cmapping)

% set planes to show 
show_planes=[8:4:39]; %[120:20:260] % axial the planes throught the T1 to be shown

% load background T1
hdr_bg=spm_vol(struct_path); % load the backgroud T1 volume; this is usually in yr directory of spm installation /canonical/
%hdr_bg= spm_vol('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/searchlight/MNI152_T1_0.5mm.nii'); % load the backgroud T1 volume;%
%%spm_vol('/afs/cbs.mpg.de/software/spm/12.7771/9.10/bionic/canonical/single_subj_T1.nii');
T1=spm_read_vols(hdr_bg); %T1
%T1(T1<17)= 0; 
T1 = T1./max(reshape(T1,1,[]));

% load overlay example
overlay= evc_mask + loc_mask*100 + ppa_mask*200;%+opa_mask*300+rsc_mask*400;
sz_overlay=size(overlay); % we need to know the size of the overlay volume for further processing

% set a color scale between 0 and the maximum in the overlay for displaying the overlay (this is just a choice we make, you can of course make a
% different choice
tmp = overlay(:);
min_val = 1; %otherwise set min val to 0 

max_val=nanmax(tmp); %find min of the overlay
color_limits= linspace(1,3,1); %scaled globally to overall min max over time
%colors=cmap(end-numel(color_limits):end,:); % that is a workaround to make the colors actually the jet colormap from half of its values...
%colors=cmap(numel(color_limits)+1:end,:); % so we end up with a color value for each of the 100 steps    

% manually set colors 
colors(1,:) = [1,0,0];%cmapping(ceil(213),:);
colors(2,:) = [0.1, 1, 0];%;cmapping(ceil(170),:);
colors(3,:) = [0.1, 0.5, 1];%cmapping(ceil(128),:);
colors(4,:) = [1 1 0];
colors(5,:) = [1 0 1];

% Creating T1 image with overlay in it. The idea is to put the overlay (which is in color) into the T1 (which is in
% grayscale)
overlayed_T1=repmat(T1,1,1,1,3); %start with the overlay T1 being equal to the T1 in 3 dimensions (it wa 1 before b?c only black and white, we need 3 dimensions for color
sz_overlayed_T1=size(overlayed_T1); % we need to know the size of the overlayed T1 for further processing

% fprintf('Size of struct image is %i %i %i', size(T1)); 
% fprintf('Size of overlay image is %i %i %i', size(overlay)); 

%clear get_out_final get_out cord_overlay_space_dump
[x y z]=meshgrid(1:sz_overlayed_T1(1),1:sz_overlayed_T1(2),1:sz_overlayed_T1(3)); % create a meshgrid over the size of the T1
zz=[x(:),y(:),z(:),ones(numel(x),1)]; % variable of x,y,z linearized and 1s (for 4D matrix transformation

cord_overlay_space_dump= mask_hdr.mat \ (hdr_bg.mat * zz'); %transformation matrix from bg to overlay, so the coordinates in T1 are set to overlay space
cord_overlay_space_dump=round(cord_overlay_space_dump); % round, as indices are full numebrs

%eliminate coords that are outside of the overlay
% get out is a funny stucture, it has rows (voxels) * columns (criteria). to exclude voxels that do not fit the criteria
get_out(1,:)=mean(cord_overlay_space_dump<1,1)>0; %index to columns that have numbers smaller than 0
get_out(2,:)=cord_overlay_space_dump(1,:)>sz_overlay(1);
get_out(3,:)=cord_overlay_space_dump(2,:)>sz_overlay(2);

get_out_final=mean(get_out,1)>0; %only those indices that fulfill all criteria

cord_overlay_space_dump(:,get_out_final)=[]; % remove impossibel coords
zz_dump=zz; zz_dump(get_out_final,:)=[]; % a new variable of meshgrid coordinates to voxels that exist

for i=1:size(zz_dump,1) %all overlay coords that fit into the anatomical
    cord_overlay_space= cord_overlay_space_dump(1:3,i); %cords in overlay space
    if overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3))>= min_val && ...
            overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3)) ~= 0 % this is my criterion- is the value there >0, You could have a very different criterion
        color=overlay(cord_overlay_space(1),cord_overlay_space(2),cord_overlay_space(3)); % a specific value in the overlay
        if color ==1; color_idx =1; elseif color==100; color_idx = 2; elseif color==200, color_idx=3; elseif color==300, color_idx=4; elseif color==400, color_idx=5; end 
        %color_code=find(abs(color_limits-color)==min(abs(color_limits-color)),1); % on a scale of 1-100, where is that wrt the min and max
        overlayed_T1(zz_dump(i,1),zz_dump(i,2),zz_dump(i,3),:) = colors(color_idx,:) ; %fill up the T1 at the location of that voxel with the color looked up
    end
end
minmax(reshape(overlayed_T1,1,[]))

% plot the T1 with the overlay on top
fh=figure('Position',[100,500,800,300],'Color','black');%[0 0 0]'
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
colormap(colors);
for i=1:numel(show_planes)
    sa(i)=subaxis(1,numel(show_planes),i,'SpacingVert',0.04 ,'SpacingHoriz', 0,'MR',0,'ML',0,'PT',0,'PB',0);
    fixed_image=rot90(squeeze(overlayed_T1(:,:,show_planes(i),:,:))); %axial slice
    %imagesc(fixed_image,'AlphaData',fixed_image(:,:,1)~=0)
    imagesc(fixed_image)
    axis equal tight off
    
    %calculate axial slice position in real world
    tmp_mat=mask_hdr.mat*[1 1 show_planes(i) 1]';
    axial_slice_nr=tmp_mat(3);
    if i == ceil(numel(show_planes)/2)
        tl(i)=title(fig_title,'FontSize',16,'VerticalAlignment','baseline','Color',[0 0 0]);
    end 
end

% add a colorbar
colormap(colors)
set(gca,'CLim',[min_val max_val])


set(sa(numel(sa)),'CLim',[min_val max_val])
cl=colorbar('southoutside')
set(cl,'Position',[0.8 0.20 0.15 0.03])
orig_ticks=get(cl,'Ticks')
new_ticks= [];
new_tick_labels = {'EVC','LOC','PPA'};
set(cl,'Ticks',new_ticks,'FontSize',14,'Color',[0 0 0]) ; %,'TickLabels',new_tick_labels
set(gcf,'color','black');
%xl =xlabel(cl,['Spearmans''s R']);
%set(xl,'FontSize',16)
%set(xl,'Position',[1.5 0.0709 0],'FontSize',16)
%cl.Label.String='R'

pic=getframe(gcf); %gat a frame of current figure
end 