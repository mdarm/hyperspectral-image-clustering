function j = jaccard(y_true, y_pred)
    % y_true: true labels
    % y_pred: predicted labels
    % j: Jaccard index
    A = sum(y_true == y_pred);
    B = sum(y_true ~= y_pred);
    C = sum(y_pred ~= y_true);
    j = A / (A + B + C);
end