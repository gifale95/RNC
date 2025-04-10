function plot_rp(cfg,i_sub,prefix,include_loc,plot_on)

% Get realignment parameters from headers of images

sub_dir = fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub));
prefix = [prefix cfg.prefix];

P = []; % all headers
ct = []; % counter of where a run ends


if include_loc
    % Localizer runs
    for i_run = 1:length(cfg.sub(i_sub).import.localizer)
        fname_path = fullfile(sub_dir,'alldata','localizer');
        fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
        P = [P; spm_vol(fname)]; % get headers
        ct = [ct length(P)];
    end
end

% Functional runs
for i_run = cfg.sub(i_sub).run_indices
    fname_path = fullfile(sub_dir,'alldata',sprintf('run%02d',i_run));
    fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
    P = [P; spm_vol(fname)]; % get headers
    ct = [ct length(P)];
end


ct = ct(1:end-1); % remove last entry because we don't need a line at the end

mparams = zeros(numel(P),12);
for i=1:numel(P),
    mparams(i,:) = spm_imatrix(P(i).mat/P(1).mat);
end

mparams = mparams(:,1:6); % reduce to realignment parameters

save(fullfile(sub_dir,'alldata','parameters',sprintf('rp_sub%02d.mat',i_sub)),'mparams','ct')


%% Plot things ( (c) Guillaume Flandin )

if ~exist('plot_on','var'), plot_on = 0; end
    
if plot_on    

% PLOT IN SPM STYLE
fg=spm_figure('GetWin','Graphics');
spm_figure('Clear','Graphics');
ax=axes('Position',[0.1 0.65 0.8 0.2],'Parent',fg,'Visible','off');
    set(get(ax,'Title'),'String','Image realignment','FontSize',16,'FontWeight','Bold','Visible','on');
    x     =  0.1;
    y     =  0.9;
    for i = 1:min([numel(P) 12])
        text(x,y,[sprintf('%-4.0f',i) P(i).fname],'FontSize',10,'Interpreter','none','Parent',ax);
        y = y - 0.08;
    end
    if numel(P) > 12
        text(x,y,'................ etc','FontSize',10,'Parent',ax); end
ax=axes('Position',[0.1 0.35 0.8 0.2],'Parent',fg,'XGrid','on','YGrid','on');
plot(mparams(:,1:3),'Parent',ax)
hold on

s = ['x translation';'y translation';'z translation'];
legend(ax, s)%0
set(get(ax,'Title'),'String','translation','FontSize',16,'FontWeight','Bold');
set(get(ax,'Xlabel'),'String','image');
set(get(ax,'Ylabel'),'String','mm');


ax=axes('Position',[0.1 0.05 0.8 0.2],'Parent',fg,'XGrid','on','YGrid','on');
plot(mparams(:,4:6)*180/pi,'Parent',ax)
s = ['pitch';'roll ';'yaw  '];
legend(ax, s)%0
set(get(ax,'Title'),'String','rotation','FontSize',16,'FontWeight','Bold');
set(get(ax,'Xlabel'),'String','image');
set(get(ax,'Ylabel'),'String','degrees');

end
