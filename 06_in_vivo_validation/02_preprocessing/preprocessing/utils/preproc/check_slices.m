function [out_im, out_slice] = check_slices(cfg,i_sub,prefix,include_loc,check_data,plot_on,maskname)

% Input variables:
%   cfg (passed)
%   i_sub: Subject number
%   prefix: Prefix of files to check (e.g. 'f')
%   check_data: if 0, don't check data
%   plot_on: if 0, don't plot
%   maskname: (optional): path to mask; don't take into account out of mask regions

%% Get volumes

out_im = []; % init
out_slice = []; % init

sub_dir = fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub));
prefix = [prefix cfg.prefix];

if check_data

P = []; % all headers
ct = []; % counter of where a run ends

if include_loc
    % Localizer runs
        fname_path = fullfile(sub_dir,'alldata','localizer');
        fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
        P = [P; spm_vol(fname)]; % get headers
        ct = [ct length(P)];
end

% Functional runs
for i_run = cfg.sub(i_sub).run_indices
    fname_path = fullfile(sub_dir,'alldata',sprintf('run%02d',i_run));
    fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
    P = [P; spm_vol(fname)]; % get headers
    ct = [ct length(P)];
end

n_runs = length(ct);
ct = ct(1:end-1); % remove last entry because we don't need a line at the end
run_ind = [1 ct+1; ct length(P)]; % where each run starts and ends

end

%% Load mask if provided

% TODO: check if correct transformations are applied to maskvol, too

if exist('maskname','var')
    maskvol = spm_read_vols(spm_vol(maskname));
else
    maskvol = ones(P(1).dim);
end

maskvol = logical(maskvol);

%% Check slices (if check_data is provided skip)

if check_data

intensity_m = cell(n_runs,1);
slicediff = cell(n_runs,1);
imdiff = cell(n_runs,1);

for i_run = 1:n_runs

    vols = P(run_ind(1,i_run):run_ind(2,i_run));
    n_vols = size(vols,1);
    if isempty(n_vols) || n_vols < 2, return, end
    V_first = vols(1);
    V_rest = vols(2:end);

    n_vols = n_vols-1;
    Hold = 0;

    [xydim n_slices] = deal(V_first.dim(1:2),V_first.dim(3));

    p1 = spm_read_vols(V_first);
    slicediff_c = zeros(n_vols,n_slices);

    intensity_m_c = zeros(n_vols,1); % Mean intensity of volume
    msg = sprintf('\n      progress:   0 percent ');
    fprintf(msg) % TODO: wrong distance
    for i_slice = 1:n_slices
        M = spm_matrix([0 0 i_slice]);
        previous_slice = p1(:,:,i_slice);
        previous_slice = previous_slice(maskvol(:,:,i_slice));
        for i_vol = 1:n_vols
            curr_slice = spm_slice_vol(V_rest(i_vol),M,xydim,Hold);
            curr_slice = curr_slice(maskvol(:,:,i_slice));
            variance = (curr_slice - previous_slice).^2;
            slicediff_c(i_vol,i_slice) = mean(variance(:));
            intensity_m_c(i_vol) = intensity_m_c(i_vol) + mean(curr_slice(:));
            previous_slice = curr_slice;
        end
        msg = sprintf('      progress:   %03d percent ',ceil(100*i_slice/n_slices)); % TODO: use display_progress
        fprintf([repmat('\b',1,length(msg)) msg]);
    end
    fprintf([repmat('\b',1,length(msg)-1) 'done...\n'])

    intensity_m_c = [mean(p1(maskvol>0)); intensity_m_c/n_slices];
    imdiff_c = [0; mean(slicediff_c,2)];
    slicediff_c = [zeros(1,n_slices); slicediff_c];

    intensity_m{i_run} = intensity_m_c;
    slicediff{i_run} = slicediff_c;
    imdiff{i_run} = imdiff_c;
    
end

intensity_m = cell2mat(intensity_m);
slicediff = cell2mat(slicediff);
imdiff = cell2mat(imdiff);

fname = sprintf('check_slices-sub%02d.mat',i_sub);
save(fullfile(sub_dir,'alldata','parameters',fname),'intensity_m','slicediff','imdiff')

end




%% Plot results

% TODO: make sure that variance is scaled *within* each run
% TODO: plot difference of realignment parameters within each run instead
% (but where?)


if plot_on

fname = sprintf('check_slices-sub%02d.mat',i_sub);
try    
    load(fullfile(sub_dir,'alldata','parameters',fname))
catch
    error('Can''t plot data when it hasn''t been calculated.')
end

% deftxti = get(0,'DefaultTextInterpreter'); % to prevent error messages
% set(0,'DefaultTextInterpreter','none');

fname = fullfile(sub_dir,'alldata','parameters',sprintf('rp_sub%02d.mat',i_sub));

if exist(fname,'file') % movement parameters exist
    mparams = [];
    ct = []; % ct: tells us where runs are separated
    load(fname); % load mparams and ct
    n_plots = 6;
else % they don't exist
    mparams = [];
    n_plots = 4;
end

n_vols = size(slicediff,1)+1;
n_slices =   size(slicediff,2);
mom = mean(intensity_m); % mom: mean of means
sslicediff = slicediff/mom; % standardized slicediff
simdiff = imdiff/mom; 

% Get outlier volumes

% use cutoff provided by cfg (this is just for marking outlier volumes)
im_cut = cfg.sub(i_sub).preproc.outlier_cutoff(1); % typical: 10
slice_cut = cfg.sub(i_sub).preproc.outlier_cutoff(2); % typical: 20
dev_cut = cfg.sub(i_sub).preproc.outlier_cutoff(3); % typical: 0.25 
ang_cut = cfg.sub(i_sub).preproc.outlier_cutoff(4); % typical: 1/120 (i.e. 0.5 arcmin)

out_im = find(simdiff>im_cut);
% remove double entries that come from abrupt changes
out_im([0; diff(out_im)]==1) = [];

out_slice = [];
for i_slice = 1:n_slices
    tmp_slice{i_slice} = find(sslicediff(:,i_slice)>slice_cut);
    if ~isempty(tmp_slice{i_slice})
    tmp_slice{i_slice}([0; diff(out_im)]==1) = [];
    for i_tmp = 1:length(tmp_slice{i_slice})
        c_slice = str2num(sprintf('%i.%02i ',tmp_slice{i_slice}(i_tmp),i_slice))';
        out_slice = [out_slice; c_slice];
    end
    end
end

out_slice = sort(out_slice);

% % find volumes that are represented at least three times and exclude them completely
% [n_out,i_out] = hist(out_slice,unique(floor(out_slice)));
% 
% % add them to out_im
% out_im = unique([out_im;i_out(n_out>=3)]);
% % remove them from out_slice
% % out_slice(ismember(floor(out_slice),out_im)) = [];
% 
% get outliers from movement parameters
out_im2 = find(any(abs(diff(mparams(:,1:3)))>dev_cut,2))+1;
% remove double entries that come from abrupt changes
 out_im2([0; diff(out_im2)]==1) = [];

out_im3 = find(any(abs(diff(mparams(:,4:6)))>ang_cut,2))+1;
out_im3([0; diff(out_im3)]==1) = [];

out_im = unique([out_im; out_im2; out_im3]);

% ssize = get(0,'screensize');
% set(gcf,'color',[1 1 1]);
% set(gcf,'position',[0.25*ssize(3)*fg-0.25*ssize(3),0.05*ssize(4),0.25*ssize(3),0.88*ssize(4)]);

figure_position = get(0,'defaultfigureposition');
figure_position = round(figure_position .* [1 1 2 1] + [-0.5*figure_position(1) -0.4*figure_position(2) 0 +0.4*figure_position(2)]); % increase width by 30% and height by 40%

figname = ['Inspection: Subject ' num2str(i_sub)];
figure('name',figname,'position',figure_position);

subplot(3,2,1);
plot(simdiff)
yval = max(simdiff);
if yval < 10, yval = 10; end
axis([0 n_vols 0 1.1*yval]);
plot_ct(ct)
hold on
if ~isempty(out_im)
    plot(out_im,1.1*max(simdiff(:)),'r*')
end
xlabel('Difference image number');
ylabel('Scaled variance');

subplot(3,2,3);
plot(sslicediff, 'x');
yrange = 1.1*max(sslicediff(:));
if yrange < 30, yrange = 30; end
axis([0 n_vols 0 yrange]);
plot_ct(ct)
hold on
for i_slice = 1:n_slices
    if ~isempty(tmp_slice{i_slice})
    plot(tmp_slice{i_slice},1.1*max(sslicediff(:)),'r*')
    end
end
xlabel('Difference image number');
ylabel('Slice by slice variance');

% % subplot(n_plots,1,3);
% figure
% plot(intensity_m/mom)
% plot_ct(ct)
% axis([0 n_vols+1 -Inf Inf]);
% xlabel('Image number');
% ylabel('Scaled mean voxel intensity');

subplot(3,2,5);
mx = max(sslicediff);
mn = min(sslicediff);
avg = mean(sslicediff);
plot(avg, 'k');
axis([0 n_slices+1 0 1.1*max(mx)]);
hold on
plot(mn, 'b');
plot(mx, 'r');
hold off
% plot_ct(ct)
xlabel('Slice number');
ylabel('Max/mean/min slice variance');

% realignment params
if ~isempty(mparams)

subplot(3,2,2); % NEW LINE!
  plot(mparams(:,1:3))
  %legend('x translation','y translation','z translation',0);
  yrange = 1.1*[min(min(mparams(:,1:3))) max(max(mparams(:,1:3)))];
  axis([0 n_vols+1 yrange(1) yrange(2)]);
  plot_ct(ct)
  hold on
  ttmp = axis; ttmp = ttmp(4);
  if ~isempty(out_im)
      plot(out_im,0.95*ttmp,'r*')
  end
  xlabel('image');
  ylabel('Translations in mm');
%   figure
% %   subplot(n_plots,1,6);
%   plot(mparams(:,4:6))
%   set(gca,'Ylim',[min(min(mparams(:,4:6))) max(max(mparams(:,4:6)))])
%   plot_ct(ct)
%   %legend('pitch','roll','yaw',0);
%   axis([0 n_vols+1 min(min(mparams(:,4:6))) max(max(mparams(:,4:6)))]);
%   xlabel('image')
%   ylabel('Rotations in degrees');

% NEW!!!

subplot(3,2,4);
hold on
%   figure
    minmin = Inf;
    maxmax = -Inf;
  for i_ct = 1:length(ct)+1
      if i_ct == 1
          plotrange_y = [zeros(1,3); diff(mparams(1:ct(i_ct),1:3))];
          plotrange_x = 1:ct(i_ct);
      elseif i_ct == length(ct)+1
          plotrange_y = [zeros(1,3); diff(mparams(ct(end)+1:end,1:3))];
          plotrange_x = ct(end)+1:length(mparams);
      else
          plotrange_y = [zeros(1,3); diff(mparams(ct(i_ct-1)+1:ct(i_ct),1:3))];
          plotrange_x = ct(i_ct-1)+1:ct(i_ct);
      end
      if minmin > min(plotrange_y(:))
          minmin = min(plotrange_y(:));
      end
      if maxmax < max(plotrange_y(:))
          maxmax = max(plotrange_y(:));
      end
      plot(plotrange_x,plotrange_y)
  end
%       plot(diff(mparams(:,1:3)))
%   set(gca,'Ylim',[minmin maxmax])
  %legend('x translation','y translation','z translation',0);
  yrange = [minmin-0.1*abs(minmin) maxmax+0.1*abs(maxmax)];
  axis([0 n_vols+1 yrange(1) yrange(2)]);
  plot_ct(ct)
  hold on
  ttmp = axis; ttmp = ttmp(4);
  if ~isempty(out_im)
      plot(out_im,0.95*ttmp,'r*')
  end
  xlabel('image');
  ylabel('Derivative of translations in mm');

  % END OF NEW!!

subplot(3,2,6);
hold on
%   figure
    minmin = Inf;
    maxmax = -Inf;
  for i_ct = 1:length(ct)+1
      if i_ct == 1
          plotrange_y = [zeros(1,3); diff(mparams(1:ct(i_ct),4:6))];
          plotrange_x = 1:ct(i_ct);
      elseif i_ct == length(ct)+1
          plotrange_y = [zeros(1,3); diff(mparams(ct(end)+1:end,4:6))];
          plotrange_x = ct(end)+1:length(mparams);
      else
          plotrange_y = [zeros(1,3); diff(mparams(ct(i_ct-1)+1:ct(i_ct),4:6))];
          plotrange_x = ct(i_ct-1)+1:ct(i_ct);
      end
      if minmin > min(plotrange_y(:))
          minmin = min(plotrange_y(:));
      end
      if maxmax < max(plotrange_y(:))
          maxmax = max(plotrange_y(:));
      end
      plot(plotrange_x,plotrange_y)
  end
%       plot(diff(mparams(:,4:6)))
  %legend('x translation','y translation','z translation',0);
  yrange = [minmin-0.1*abs(minmin) maxmax+0.1*abs(maxmax)];
  axis([0 n_vols+1 yrange(1) yrange(2)]);
  plot_ct(ct)
  hold on
  ttmp = axis; ttmp = ttmp(4);
  if ~isempty(out_im)
      plot(out_im,0.95*ttmp,'r*')
  end
  xlabel('image');
  ylabel('Derivative of rotations in mm');  
  
end

% and label with first image at bottom
% cp = get(gca,'Position');
% wfa =  axes('Position', [0 0 1 1], 'Visible', 'off');
% img1 = deblank(vols(1,:));
% text(0.5,cp(2)/2.5,{'First image:',img1},'HorizontalAlignment','center');

% set(0,'DefaultTextInterpreter',deftxti); % set back to original

end

%------------------------
function plot_ct(ct)
hold on
% xlim = get(gca,'Xlim');
ylim = get(gca,'Ylim');
for i = 1:length(ct)
    plot([ct(i)+0.5 ct(i)+0.5],ylim,'.-g')
end
hold off