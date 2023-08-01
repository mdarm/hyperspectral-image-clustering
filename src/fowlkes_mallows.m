function fm = fowlkes_mallows(y_true, y_pred)
    % y_true: true labels
    % y_pred: predicted labels
    % fm: Fowlkes-Mallows index
    n = length(y_true);
    true_clusters = unique(y_true);
    pred_clusters = unique(y_pred);
    n_true = length(true_clusters);
    n_pred = length(pred_clusters);
    true2pred = zeros(n_true, n_pred);
    pred2true = zeros(n_pred, n_true);
    for i = 1:n_true
        for j = 1:n_pred
            true2pred(i,j) = sum((y_true == true_clusters(i)) & (y_pred == pred_clusters(j)));
            pred2true(j,i) = true2pred(i,j);
        end
    end
    fm = sqrt((sum(sum(true2pred.^2))/sum(sum(pred2true.^2)))*(sum(sum(pred2true.^2))/sum(sum(true2pred.^2))));
end
