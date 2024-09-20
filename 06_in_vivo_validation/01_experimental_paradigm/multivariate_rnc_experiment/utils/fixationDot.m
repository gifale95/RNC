function fixationDot(win, screenRect, fixation_size_pixels)
% A small semi-transparent red fixation dot with a black border
% (0.2° × 0.2°, 50 opacity) presented at the center of the stimuli.

	% Get the centre coordinate of the window
	[xCenter, yCenter] = RectCenter(screenRect);

	% Fixation dot screen rectangle
	rect = [xCenter-fixation_size_pixels/2, yCenter-fixation_size_pixels/2, ...
		xCenter+fixation_size_pixels/2, yCenter+fixation_size_pixels/2];

	% Draw the fixation dots
	Screen('FillOval', win, [255 0 0 128], rect);
	Screen('FrameOval', win, [0 0 0 128], rect, 2);
	
end