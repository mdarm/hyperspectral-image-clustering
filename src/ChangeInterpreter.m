function ChangeInterpreter(h, Interpreter)
% ChangeInterpreter() changes the interpreter of figure h.

    % Find all string type objects
    TexObj = findall(h, 'Type', 'Text');
    LegObj = findall(h, 'Type', 'Legend');
    AxeObj = findall(h, 'Type', 'Axes');  
    ColObj = findall(h, 'Type', 'Colorbar');
    
    Obj = [TexObj; LegObj]; % Tex and Legend opbjects can be treated similarly    
    n_Obj = length(Obj);
    for i = 1:n_Obj
        Obj(i).Interpreter = Interpreter;
    end
    
    Obj = [AxeObj; ColObj]; % Axes and ColorBar objects can be treated similarly 
    n_Obj = length(Obj);
    for i = 1:n_Obj
        Obj(i).TickLabelInterpreter = Interpreter;
    end
end