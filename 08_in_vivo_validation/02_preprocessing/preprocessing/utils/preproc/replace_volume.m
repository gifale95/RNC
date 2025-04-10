function replace_volume(cfg,i_sub,prefix,include_loc)

% Instead, move original files to the new corrupted folder (with
% renaming with c) and overwrite original

% Get headers

disp('Running replace_volume...')

P = select_files_adapted(cfg,i_sub,prefix,include_loc,0);

artifact = cfg.sub(i_sub).preproc.corrupted_volumes(:);
artifact = artifact(artifact==floor(artifact)); % keep only those where the entire volume should be replaced

prev_n_vols = 0;
for i_run = 1:length(P)
    
    n_vols = length(P{i_run});
    
    % check if there are any values to interpolate
    x_vals = (1:n_vols)' + prev_n_vols; % we keep counting volumes across runs, i.e. we gotta correct for that
    rm_vals = intersect(x_vals,artifact);
    
    if isempty(rm_vals)
        % Do not execute if no volumes need replacement
        prev_n_vols = prev_n_vols + n_vols;
        continue
    end
    
    % If there are any, get important values from artifact
    rm_ind = rm_vals - prev_n_vols;

    % only load four volumes around the current rm_ind (because this
    % should get everything captured by cubic interpolation anyway)
    fprintf('Getting header of run %i\n',i_run)
    Pmat = cell2mat(P{i_run});
    load_ind = bsxfun(@plus,rm_ind,(-4:4));
    load_ind = unique(load_ind(:));
    load_ind = max(load_ind,1);
    load_ind = min(load_ind,n_vols);
    hdr = spm_vol(Pmat(load_ind,:));

    
    
    % get mask volume from first volume
    maskvol = spm_read_vols(spm_vol(Pmat(1,:)));
    maskvol = double(maskvol>0.6*mean(maskvol(:)));
    % dilate three times
    for i = 1:3
        maskvol = spm_dilate(maskvol);
    end
    
    mask_index = find(maskvol);
    
    % Get all voxels in mask
    [x y z] = ind2sub(hdr(1).dim(1:3),mask_index);
    
    % Memory map volumes
%     disp('Memory mapping volumes...')
    nc_vols = length(load_ind);
    vol = zeros([length(mask_index) nc_vols]);
    msg_length = [];
    msg = 'Memory mapping volumes...';
    for i_vol = 1:nc_vols
        vol(:,i_vol) = spm_sample_vol(hdr(i_vol),x,y,z,0);
        msg_length = display_progress_ana(i_vol,nc_vols,msg_length,msg);
    end
    disp('done.')
    
    % remove indices where no value exceeds 100
%     keep_ind = find(all(vol>100,2));
%     vol = vol(keep_ind,:);
%     mask_index = mask_index(keep_ind);
    
    % now temporally interpolate over remaining voxels
    [keep_ind,keep_indind] = setdiff(load_ind,rm_ind);
    [tmp,rm_indind] = intersect(load_ind,rm_ind);
    
    % Carrying out temporal interpolation
    disp('Running temporal interpolation...')
    yy = spline(keep_ind,vol(:,keep_indind),rm_ind);
    disp('done.')
    
    % Replace volumes
    vol(:,rm_indind) = yy;
    
    
    msg_length = [];
    msg = 'Writing replaced volumes...';
    
    new_dir = fullfile(cfg.sub(i_sub).dir,'alldata','corrupted');
    if ~isdir(new_dir), mkdir(new_dir), end
    
    for i_vol = 1:n_vols
        
        if any(rm_ind==i_vol)
        
        orig_fname = deblank(Pmat(i_vol,:));
        [fp fn fext] = fileparts(orig_fname);
        if strcmp(fext(end-1:end),',1')
            fext = fext(1:end-2);
        end
        orig_fname = fullfile(fp,[fn fext]);
        new_fname = fullfile(new_dir,[fn fext]);
                
        h(1) = copyfile(orig_fname,new_fname);
        if strcmpi(fext,'.img')
            h(2) = copyfile(fullfile(fp,[fn '.hdr']),fullfile(new_dir,[fn '.hdr']));
        end
        
        if any(h==0)
            error('Could not replace volume')
        end
        
        curr_hdr = spm_vol(Pmat(i_vol,:));
        %if the dimensions in the bad volume dont match with the other
        %volumes then take the volume before the current volume for rewriting
        if any(hdr(1).dim ~= curr_hdr.dim)
            curr_hdr = spm_vol(Pmat(i_vol-1,:));
        end 
        orig_vol = spm_read_vols(curr_hdr);
        orig_vol(mask_index) = vol(:,rm_ind==i_vol); % overwrite
        
        if i_vol ~= 1
            curr_hdr = spm_vol(Pmat(i_vol-1,:)); % replace by previous header if possible
        end
        
        curr_hdr.fname = orig_fname;
        spm_write_vol(curr_hdr,orig_vol);
               
        end
        
        msg_length = display_progress_ana(i_vol,n_vols,msg_length,msg);

        
    end
    
    prev_n_vols = prev_n_vols+n_vols; % update for next loop
    
end