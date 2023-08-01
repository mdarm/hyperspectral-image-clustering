%% Clustering Algorithms, Project
%  Image Clustering on the Salinal valley data.
%
%  This script makes use of the following provided functions:
%
%     rand_data_init.m
%     k_means.m
%     possibi.m
%
%  Two further functions, written by the authors, were also used
%  for editting plot variables, namely:
%
%     PlotDimensions.m
%     ChangeInterpreter.m
%
%  In addition, many other standard MATLAB functions were
%  also utilized throughout the script to aid in the clustering
%  process.

%% Importing and processing data
clear
format compact
close all

load Salinas_Data

% Size of the Salinas cube
[p, n, l] = size(Salinas_Image);

% Making a two dimensional array whose rows correspond to the
% pixels and the columns to the bands, containing only the pixels
% with nonzero label.
X_total = reshape(Salinas_Image, p*n, l);
L       = reshape(Salinas_Labels, p*n, 1);

% This contains 1 in the positions corresponding to pixels
% with known class label.
existed_L = (L > 0);
X         = X_total(existed_L, :);

% The class labels.
y = L(existed_L);

% Abbreviations px = no. of rows (pixels) and nx = no. of columns
% (bands).
[px, nx] = size(X);

% Normalize each column of the dataset to its mean
% and also normalise to it standard deviation.
mean_X = mean(X);
s      = std(X);
X      = X - mean_X ./ s;

% Find the optimal number of components to retain.
[coeff, score, latent, tsquared, explained] = pca(X);
explained_variance = 0;
k = 0;

while explained_variance < 99
    k = k + 1;
    explained_variance = explained_variance + explained(k);
end


% Print original datase alongside the recundstructed dataset
% in order to assess reconstructibility (explained variance 
% does not necessarilly imply reconstructibility).
figure; biplot(coeff(:,1:k), 'scores', score(:,1:k));


% Reduce the dataset to the optimal number of components
X = score(:, 1:k);

% Find the optimal number of clusters using the Calinski-Harabasz
% criterion and the 'elbow' method.
myfunc = @(X,K)(kmeans(X, K, 'Distance', 'cityblock',...
                             'MaxIter', 1000));
evaluation = evalclusters(X, myfunc, "CalinskiHarabasz",...
                          "KList", 1:20);
figure; plot(evaluation);

% Initialize the vector to store the within-cluster
% sum of squares (WCSS).
wcss = zeros(1, 20);

options = statset('UseParallel', 1, 'MaxIter', 1000);
for i = 1:20
    % Perform k-means clustering
    [~, ~, sumd] = kmeans(X, i, 'Distance', 'cityblock', 'Options', options);
    
    % Store the WCSS for each value of k
    wcss(i) = sum(sumd);
end

% Plot the WCSS against the number of clusters.
figure;
axes1 = axes('Parent', gcf);
hold(axes1, 'on');

% Create line.
line(1:20, wcss, 'Parent', axes1, 'MarkerSize', 10, 'Marker', '.');

% The 'elbow' point is the point at which the WCSS starts to plateau
% as the optimal number of clusters, as it indicates that adding more
% clusters will not significantly improve the WCSS.

% Calculate the second derivative of the WCSS
d2 = diff(diff(wcss));

% Find the local maxima of the second derivative
[~, idx] = findpeaks(-d2);

% The optimal number of clusters is the number of clusters where the second
% derivative changes sign
k1 = idx(1) + 2;
plot(k1, wcss(k1), 'Marker', 'o', 'Color', [0 0 1]);

xlabel('Number of Clusters');
ylabel('Within-Cluster Sum of Squares');
box(axes1,'on');
hold(axes1,'off');

% Prompt optimal number of clusters as calculated by using the
% Calinski-Harabasz criterion and the Elbow method.
prompt = sprintf('Enter the number of clusters k (Calinski-Harabasz criterion = %d, Elbow Method = %d): ', evaluation.OptimalK, k1);
k = input(prompt);

%% Cost Function Optimisation Algorithms

% K-means
options     = statset('UseParallel', 1, 'MaxIter', 1000);
num_repeats = 100;
min_sumd    = inf;
for i = 1:num_repeats
    [idx,C,sumd,D] = kmeans(X, k, 'Start', 'sample', 'Options', options);
    if min_sumd > sum(sumd)
        min_sumd = sum(sumd);
        cl_label = idx;
    end
end

cl_label_tot           = zeros(p*n, 1);
cl_label_tot(existed_L)= cl_label;
im_cl_label            = reshape(cl_label_tot, p, n);

% Fuzzy C-means
num_repeats = 2;
min_cost    = inf;
options     = [2; 1000; 1e-5; 1];
fuzzifier_values = 1.1:0.1:2;

for i = 1:numel(fuzzifier_values)
    options(1) = fuzzifier_values(i);
    for j = 1:num_repeats
        [center, U, obj_fcn] = fcm(X, k, options);
        if min_cost > min(obj_fcn)
            min_cost = obj_fcn;
            [~, cl_label1] = max(U, [], 1);
            U1 = U;
        end
    end
end

cl_label_tot            = zeros(p*n,1);
cl_label_tot(existed_L) = cl_label1;
im_cl_label2            = reshape(cl_label_tot,p,n);

% Possibilistic C-means
eta = ones(1,k);  % Eta parameters of the clusters
q = 40;           % q parameter of the algorithm
sed = 1;          % Seed for random generator
init_proc = 2;    % Use "rand_data_init" initialization procedure
e_thres = 0.0001; % Threshold for termination condition

% Run the possibilistic clustering algorithm
[U, theta] = possibi(X', k, eta, q, sed, init_proc, e_thres);

% Assign each data point to the cluster with the highest compatibility
[~, cl_label2] = max(U, [], 2);

cl_label_tot(existed_L)=cl_label2;
im_cl_label3=reshape(cl_label_tot,p,n);


% Probabilistic
ks       = 2:1:12;
options  = statset('MaxIter', 1000);
best_BIC = inf;
for i = 1:length(ks)
    k = ks(i);
    options.Start = 'kmeans';
    gm  = fitgmdist(X, k, 'CovarianceType', 'full', 'Options', options);
    BIC = gm.BIC;
    if BIC < best_BIC
        best_BIC   = BIC;
        best_model = gm;
    end
    
    options.Start = 'kmeans';
    gm = fitgmdist(X,k,'CovarianceType','diagonal','Options',options);
    BIC = gm.BIC;
    if BIC < best_BIC
        best_BIC   = BIC;
        best_model = gm;
    end
end

cl_label3               = cluster(best_model, X);
cl_label_tot            = zeros(p*n, 1);
cl_label_tot(existed_L) = cl_label3;
im_cl_label4            = reshape(cl_label_tot, p, n);

%% Hierarchical Algorithms

% Metric pool from which hierachical algorithms will choose from.
metrics = {'euclidean', 'cityblock', 'mahalanobis', 'chebychev'};

% Complete linkage.
max_coph_corr = -inf;
best_metric = '';
best_Z = [];

for i = 1:length(metrics)
    Y = pdist(X, metrics{i});
    Z = linkage(Y, 'complete');
    coph_corr = cophenet(Z, Y);
    if coph_corr > max_coph_corr
        max_coph_corr = coph_corr;
        best_metric   = metrics{i};
        best_Z = Z;
    end
end

fprintf('The best metric, for complete linkage, is %s with a cophenetic correlation coefficient of %f\n', best_metric, max_coph_corr);
figure;

cl_label4               = cluster(best_Z, 'maxclust', k);
cl_label_tot            = zeros(p*n, 1);
cl_label_tot(existed_L) = cl_label4;
im_cl_label5            = reshape(cl_label_tot, p, n);

% Print dendogram.
figure;
dendogram(Z, 0);
set(gca, 'xticklabel', []);


% WPGMC linkage.
Y  = pdist(X, metrics{1});
Z1 = linkage(Y, 'WPGMC');

cl_label5               = cluster(Z1, 'maxclust', k);
cl_label_tot            = zeros(p*n, 1);
cl_label_tot(existed_L) = cl_label5;
im_cl_label6            = reshape(cl_label_tot, p, n);

% Print dendogram.
figure;
dendogram(Z1, 0);
set(gca, 'xticklabel', []);


% Ward linkage.
Y  = pdist(X, metrics{1});
Z2 = linkage(Y, 'ward');

cl_label6               = cluster(Z2, 'maxclust', k);
cl_label_tot            = zeros(p*n, 1);
cl_label_tot(existed_L) = cl_label6;
im_cl_label7            = reshape(cl_label_tot, p, n);

% Print dendogram.
figure;
dendogram(Z2, 0);
set(gca, 'xticklabel', []);

%% Qualitative Evaluation

% Plot the reconstructed images for qualitative evaluation.
figure;

subplot(4,2,1);
imagesc(Salinas_Labels); axis off;
title("Original representation");

subplot(4,2,2);
imagesc(im_cl_label); axis off;
title('K-means');

subplot(4,2,3);
imagesc(im_cl_label2); axis off;
title('Fuzzy C-means');

subplot(4,2,4);
imagesc(im_cl_label3); axis off;
title('Possibilistic C-means');

subplot(4,2,5);
imagesc(im_cl_label4); axis off;
title('Probabilistic');

subplot(4,2,6);
imagesc(im_cl_label5); axis off;
title('Complete-link');

subplot(4,2,7);
imagesc(im_cl_label6); axis off;
title('WPGMC');

subplot(4,2,8); 
imagesc(im_cl_label7); axis off;
title('Ward');

% Perform PCA.
[coeff, score] = pca(X);

% Plot the first two principal components colored by the original labels.
figure;
scatter(score(:,1), score(:,2), [], y, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Original Figure');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by K-means.
figure;
scatter(score(:,1), score(:,2), [], cl_label, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with K-means Clustering');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Fuzzy C-means.
figure;
scatter(score(:,1), score(:,2), [], cl_label1', 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Fuzzy C-means Clustering');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Possibilistic C-means.
figure;
scatter(score(:,1), score(:,2), [], cl_label2, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Possibilistic C-means Clustering');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Probabilistic Clustering.
figure;
scatter(score(:,1), score(:,2), [], cl_label3, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Probabilistic Clustering');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Hierarchical Clustering (complete linkage).
figure;
scatter(score(:,1), score(:,2), [], cl_label4, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Hierarchical Clustering (complete linkage)');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Hierarchical Clustering (WPGMC linkage).
figure;
scatter(score(:,1), score(:,2), [], cl_label5, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Hierarchical Clustering (WPGMC linkage)');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Plot the first two principal components colored by the cluster labels
% obtained by Hierarchical Clustering (ward linkage).
figure;
scatter(score(:,1), score(:,2), [], cl_label6, 'filled', 'MarkerEdgeColor', 'k');
title('PCA result with Hierarchical Clustering (ward linkage)');
xlabel('First Principal Component');
ylabel('Second Principal Component');
colorbar()

% Configure and save all figures at the end by using a loop to iterate
% over all figure objects.
for i = 1:numel(findobj('type', 'figure'))
    PlotDimensions(figure(i), 'centimeters', [15.747, 16], 12);
    ChangeInterpreter(figure(i), 'LaTeX');
    exportgraphics(gcf, sprintf('plot%d.pdf', i), 'BackgroundColor', 'none');
end

% Gather all the clustering labels.
cl_labels = {cl_label, cl_label1,...
             cl_label2, cl_label3,...
             cl_label4, cl_label5,...
             cl_label6};

%% Quantitative Evaluation

% Initialise evaluation indices.
ARI = zeros(1, length(cl_labels));
NMI = zeros(1, length(cl_labels));
for i = 1:length(cl_labels)
    % Adjusted Rand Index (ARI).
    ARI(1, i) = rand_index(y, cl_labels{i}, 'adjusted');
    
    % Normalised Mutual Information (NMI).
    NMI(1, i) = nmi(y, cl_labels{i});
end

% Print the quantitative evaluation results in a table T.
clustering_methods = {'K-means', 'Fuzzy C-means',...
                      'Possibilistic C-means', 'Probabilistic',...
                      'Hierarchical (complete)',...
                      'Hierarchical (WPGMC)', 'Hierarchical (ward)'};
                  
T = table(clustering_methods', ARI', NMI',...
          'VariableNames', {'Clustering_Method',...
          'Adjusted Rand Index', 'Normalised Mutual Information'});