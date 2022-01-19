This repository contains an implementation of a clustering technique whose structure is similar to that of an unsupervised random forest; it is based on the Mondrian stochastic process and it consists of a hierarchical partition of the space of definition of the given dataset. 

The whole code consists of six files:
- Mondrian.py
- Metrics.py
- Matrix.py
- Partitioning.py
- Merging.py
- Plot.py
 
Mondrian.py contains the main functions used to apply the clustering tree and forest to a given dataset; it imports all the other files, except Plot.py. This latter file allows to visualize the space and dataset clusterization. 

The *example* folder contains two .ipynb files, showing two applications of the clustering algorithm to 2 and 3 dimension datasets.

# Brief description of the code

## Matrix.py
It consists of the function *distance_matrix*. It takes as input the array of the initial dataset and gives as output three dataframes: 
- *data_index* stores the indexed samples
- *cut_matrix* stores the hyperplanes associated to each pair of samples (each hyperplane is characterized by its normal vector coordinates and magnitude and an index)
- *point_cut_distance* stores the distances between all the samples and each hyperplane (each columns corresponds to a hyperplane and each row to a sample)


## Partitioning.py
It corresponds to the first phase of the clustering tree algorithm, where the dataset and its underlying space are recursively partitioned; it consists of five functions:


### *recursive_process*
It takes as input a polytope and the subset of data belonging to it and divides them into two parts giving as output the two new polytopes, the corresponding subsets of data and the time in which the cut is generated.
The other input parameters are the initial time, the lifetime of the process, the distance matrix computed through the *distance_matrix* function, the string identifying the similarity metric and the value of the exponent at which the metric is raised.
The function performs the following steps:
- it generates the time of the cut
- it reduces the distance matrix only to the points contained in the input polytope
- it generates the cutting hyperplane through the *cut_choice* function
- it determines the two new polytopes through the *space_splitting* function
- it determines the two new subsets of data and associates them to the corresponding polytopes through the *data_assignment* function

It doesn't perform the split if the number of points contained in the polytope is less or equal than two or if the time of the cut is higher than the lifetime.

### *cut_choice*
It takes as input the subset of data that will be splitted, the distance matrix restricted to the subset, the string identifying the similarity metric and the value of the exponent at which the metric is raised.
For each possible cutting hyperplane (whose information is stored in the *distance_matrix* output), the metric, that evaluates the similarity of the two groups of points divided by the hyperplane, is computed. Five different metrics are considered and the chosen one is determined by the input string parameter, that can be set equal to of the following strings: 'variance', 'centroid_diff', 'centroid_ratio', 'min' and 'min_corr'.
The cutting hyperplane is extracted with a probability proportional to the metric value and the function gives as output the orthogonal single norm vector identifiying the hyperplane, its distance from the origin of the axes and the restriction of the distance matrix containing the information of the distances between the hyperplane and all the points belonging to the subset of data given as input.

### *space_splitting*
It takes as input the polytope that will be splitted and the two quantities that identify the cutting hyperplane and are obtained as output of the previously described function. It gives as output the two new polytopes, obtained by respectively intersecting the father space with one of the two half spaces created by the cutting hyperplane. 

### *data_assignment*
It takes as input the two new polytopes, obtained by the *space_splitting* function, and the restriction of the distance matrix obtained as output of the *cut_choice* function. It divides the data into two subsets, according to the sign of their distances from the hyperplane, and assigns each of them to the polytope in which are contained.

### *partitioning*
Given as input the dataset that has to be clustered, in addition to the other characteristic parameters of the process, it allows to iterate the *recursive_process* function, obtaining the hierarchical partitioning of the dataset and its underlying space. The single cutting process is associated to each element of the list *m*; each element identifies a polytope, the data contained in it and the time point at which the polytope has been created. The polytope and data corresponding to each element of the list are splitted into two parts through the *recursive_process* function and, after each split, the new pairs of polytopes/data are added to the list.
The function gives as output the list *m* and a dataframe containing the information about the polytopes that are hierarchically created, their characteristic numbers, their creation time, the father polytopes from which are generated and if they are leaves of the corresponding tree structure.  



## Merging.py
It corresponds to the second phase of the clustering tree algorithm, where the neighboring leaf subspaces obtained from the partitioning phase are progressively merged on the basis of their similarity; it consists of five functions:

### *merging*
It takes as input the outcome of the *partitioning* function and the string identifying the chosen metric. It performs the following steps:
- it associates the neighboring subspaces to each leaf polytope, through the *neighbors* function
- it merges the polytopes containing only one point with the nearest subspace, through the *merge_single_data* function.
- it iteratively computes the metric for each pair of neighboring polytopes, through the *polytope_similarity* function, and merges the two nearest neighbors, through the *merge_two_polytopes* function; the iterative process stops when all the polytopes are merged.
 
It gives as output two lists, with each element describing the division of the initial space/dataset after each iteration. Each element of the first list stores the characteristic numbers of the subspaces in which the initial space is divided and, for each subspace, it associates the neighboring polytopes and the subspaces that have been merged to the considered one in order to obtain the current space partition. Each element of the second list stores the data contained in each subspace considered in the previous list.


### *neighbors*
Given as input the hierarchy of polytopes obtained from the *partitioning* function, it determines which leaf subspaces are neighbors.

### *merge_two_polytopes*
It takes as input the objects, that describe the current partitioning of the space (and will be stored as elements of the the lists given as output of the *merging* function), and the characteristic numbers of the two subspaces that have to be merged. It merges the two subspaces and gives as output the updated input objects.

### *merge_single_data*
It applies the *merge_two_polytopes* function to each polytope containing a single sample and its nearest neighboring subspace.

### *polytope_similarity*
It calculates the similarity metric for each pair of neighboring polytopes. The two subspaces with the lower value of the metric are merged in the same iteration.

## Metrics.py

### *variance_metric*, *centroid_metric*, *min_dist_metric*
They compute the similarity metric (defined in different ways), given as input two datasets.

### *compute_metric*
Given as input two datasets and the string identifying the metric ('variance', 'centroid_diff', 'centroid_ratio', 'min' and 'min_corr'), it calculates the corresponding metric through the previously defined functions. It is imported in Partitioning.py and Merging.py. 


## Mondrian.py

### *mondrian_tree*
It executes the *distance_matrix* , *partitioning* and *merging* functions and gives as outputs the outcomes of the two clustering phases.

### *class_assignment*
It assigns a label to the clustered data, for each possible configuration of the space partitioning obtained as output of the tree (end of merging phase). 

### *ami* 
It takes as input a list of sets of labels. Each element of the list is the output of the *class_assignment* function and corresponds to a specific tree result. For each possible number of clusters/polytopes in which the space is divided, the adjusted mutual information is computed for each pair of tree outcomes. It gives as output the list of AMI coefficients, their mean and their standard deviation.

### *mondrian_forest*
Given as input the number of trees that will constitute the forest, it executes the *distance_matrix* function and iterates the *partitioning*, *merging* and *class_assignment* functions. Then it computes the AMI coefficients through the *ami* function and gives as output the partitioning, merging outcome for each tree and the AMI coefficients.

### *save_tree* and *save_forest*

### *read_tree* and *read_forest*
They read the .json/.txt files storing the tree/forest outcome.

## Plot.py

### *compute_vertices*
Given as input a polytope object, gives as otput its vertices (they are required by the plotting functions).

### *plot2D_partitioning*
Plot of the space partitioning obtained after the first phase of the tree algorithm, in case of 2 dimension datasets.

### *plot2D_merging*
Plot of the space classification obtained as result of the tree, in case of 2 dimension datasets; the number of clusters in which the dataset is divided is requires as input parameter.

### *plot3D*
Plot of the classification obtained as result of the tree, in case of 3 dimension datasets; the number of clusters in which the dataset is divided is requires as input parameter. The data and space classification are separately shown.

### *plot_AMI*
Plot of the averaged adjusted mutual information vs the number of clusters.
