function [msg_length] = display_progress_ana(i,n,msg_length,msg)

if ~exist('msg','var')
    msg = '';
end

if i == 1, fprintf('%s\n',msg), end

if i == 1 || mod(i,ceil(n/100)) == 0 || i == n

    message = sprintf('Percent completed: %3i',round(100*i/n));
    
    % delete old message
    if ~isempty(msg_length)
        fprintf(repmat('\b', 1, msg_length)); % delete old text
    end

        msg_length = length(message);

    % print message
    fprintf(message)
    
end
    
if i == n, fprintf('\n'), end