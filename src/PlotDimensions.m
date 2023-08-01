function PlotDimensions(h, Units, Plotsize, Fontsize)
% PlotDimensions() changes the string units, the fontsize
% and the unit size of the figure h.

    h.Units = Units; % measurement units
    h.Position(2) = (h.Position(2) - 8.5); % bottom-left corner of plot 
    h.Position((3:4)) = Plotsize; % usually [15.747, 9] 
    set(findall(h, '-property', 'FontSize'), 'FontSize', Fontsize); % fontsize
end 
