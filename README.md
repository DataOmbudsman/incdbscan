# IncrementalDBSCAN

`incdbscan` is an implementation of IncrementalDBSCAN, the incremental version of the DBSCAN clustering algorithm.

IncrementalDBSCAN lets the user update the clustering by inserting or deleting data points. The algorithm yields the same result as DBSCAN but without reapplying DBSCAN to the modified data set.

Thus, IncrementalDBSCAN is ideal to use when the size of the data set to cluster is so large that applying DBSCAN to the whole data set would be costly but for the purpose of the application it is enough to update an already existing clustering by inserting or deleting some data points.

The implementation is based on the following paper. To see what's new compared to the paper, jump to [Notes on the IncrementalDBSCAN paper](#notes-on-the-incrementaldbscan-paper).

> Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Wimmer, Michael; Xu, Xiaowei (1998). *Incremental Clustering for Mining in a Data Warehousing Environment.* In: Proceedings of the 24rd International Conference on Very Large Data Bases (VLDB 1998).

<p align="center">
  <img src="./images/illustration_circles.gif" alt="indbscan illustration">
</p>

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#Performance)
- [Notes on the IncrementalDBSCAN paper](#notes-on-the-incrementaldbscan-paper)

# Installation

TODO

# Usage

The algorithm is implemented in the `IncrementalDBSCAN` class.

There are 3 methods to use:
- `insert` for inserting data points into the clustering
- `delete` for deleting data points from the clustering
- `get_cluster_labels` for obtaining cluster labels

All of the methods take a batch of data points in the form of an array of shape `(n_samples, n_features)` (similar to the `scikit-learn` API).

```python
from sklearn.datasets import load_iris
X = load_iris()['data']
X_1, X_2 = X[:100], X[100:]

from incdbscan import IncrementalDBSCAN
clusterer = IncrementalDBSCAN(eps=0.5, min_pts=5)

# Insert 1st batch of data points and get their labels
clusterer.insert(X_1)
labels_part1 = clusterer.get_cluster_labels(X_1)

# Insert 2nd batch and get labels of all points in a one-liner
labels_all = clusterer.insert(X_2).get_cluster_labels(X)

# Delete 1st batch and get labels for 2nd batch
clusterer.delete(X_1)
labels_part2 = clusterer.get_cluster_labels(X_2)
```

For a longer description of usage check out the [notebook](./notebooks/incdbscan-usage.ipynb) developed just for that!

# Performance

Cluso 8k/10k dataset: insertion chart.
- Mention per-datapoint cost as well.
- Mention batch inserting could be faster.
Cluso 8k/10k dataset: deletion chart. Bottlenecks.
- Mention why it's slow.

# Notes on the IncrementalDBSCAN paper
The work by Ester et al. 1998 lays the groundwork for this implementation of IncrementalDBSCAN. However, some parts of the algorithm are not covered in the paper. In this section, these holes will be identified, and solutions are proposed to fill them.

Notations used:
- *D*: the set of data objects.
- *N<sub>Eps</sub>(p)*: the set of all objects that are in the *Eps*-neighborhood of *p*.
- *UpdSeed<sub>Ins</sub>*, the set of update seeds after insertion, is defined in *Definition 7* as the set of core objects in the *Eps*-neighborhood of those objects that gain their core object property as a result of the insertion into *D*.
- *UpdSeed<sub>Del</sub>*, the set of update seeds after deletion, is defined in *Definition 7* as the set of core objects in the *Eps*-neighborhood of those objects that lose their core object property as a result of the deletion from *D*.

## Absorption when *UpdSeed<sub>Ins</sub>* is empty
Let's suppose that cluster *C* is already established, and a new object *p* is inserted in the *Eps*-neighborhood of a core object *c* of cluster *C*. Additionally, suppose that there are not enough objects in *N<sub>Eps</sub>(p)* for *p* to become a core object and that no other objects become core objects due to the insertion.

Now, how should *p* be handled?

Since there are no new core objects after the insertion, *UpdSeed<sub>Ins</sub>* is empty. According to *Section 4.2*, _"if *UpdSeed<sub>Ins</sub>* is empty [...] then *p* is a noise object."_ But we also know that *p* is in *N<sub>Eps</sub>(c)*, and according to *Definition 4* this means that it should be assigned to cluster *C*. There is clearly a contradiction here.

**Solution**: In this implementation, even if *UpdSeed<sub>Ins</sub>* is empty, *p* is assigned to cluster *C* if *c* is a core object of cluster *C* and *p* is in *N<sub>Eps</sub>(c)*.

## Simultaneous creations, absorptions and merges
In *Section 4.2*, cases of creation, absorption and merge are presented. These are indeed essential building blocks of IncrementalDBSCAN. However, the paper fails to mention that these events can happen simultaneously.

Let's see an example. Suppose we have a 1 dimensional data set with 6 objects (*a*, *b*, *c*, *x*, *y*, *z*) as illustrated in the following block. The coordinates of the objects are as noted below their names.
<pre>
- - - c - b - a - - - - - - - x - y - z - - -
     -4  -3  -2               2   3   4      
</pre>

If we apply IncrementalDBSCAN to the data set with *Eps*=2 and *MinPts*=4, no clusters are created since none of the objects have an *Eps*-neighborhood that contain at least 4 objects. That is, all objects are noise objects.

We now insert object *p* at position 0.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

After the insertion, both *N<sub>Eps</sub>(a)* and *N<sub>Eps</sub>(x)* contain 4 objects, so *a* and *x* become core objects. *UpdSeed<sub>Ins</sub>* then contains the new core objects, *a* and *x*. According to the paper if _"UpdSeed<sub>Ins</sub> contains only core objects which did not belong to a cluster before the insertion of p, i.e. they were noise objects or equal to p, [...] a new cluster containing these noise objects as well as p is created."_

Here *UpdSeed<sub>Ins</sub>* contains only new core objects (*a* and *x*) but all 7 of the objects cannot be part of one cluster, since not all objects would be density-reachable from any other object in the cluster (because, e.g., *a* is not directy density-reachable from *p*). Thus, the definition of a cluster (*Definition 4*) wouldn't hold. This is contradictory to the above quote from *Section 4.2*.

Analogous examples can be constructed for absorptions and merges. E.g., a creation and an absorption can happen at the same time, or even two merges can. But the paper doesn't cover these cases.

**Solution**: *UpdSeed<sub>Ins</sub>* should be broken down to components in which each object is density-connected to any other object in the component. The rules of creation, absorption and merge should be applied not to *UpdSeed<sub>Ins</sub>* as a whole but to each component individually.

## Extended definition of *UpdSeed<sub>Del</sub>*

The point of defining *UpdSeed<sub>Del</sub>* is to take the first step towards finding all objects in the whole object set that eventually might be affected by a deletion. *UpdSeed<sub>Del</sub>* contains the _"seed objects for the update"_.

Let's take the following object set *D* of 7 one dimensional objects (*a*, *b*, *c*, *p*, *x*, *y*, *z*). The coordinates of the objects are as noted below their names.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

Suppose we apply IncrementalDBSCAN to the objects with *MinPts*=3 and *Eps*=2. As a result, all objects belong to a single cluster.

Now suppose we delete *p*. Following *Definition 7* in the paper, *UpdSeed<sub>Del</sub>* would be empty, since there is no object that is core in *D* but not in *D* \ {*p*}. Thus, according to the definition, there are no seed objects for the update.

This is in conflict with the results of the deletion, in which there are now two clusters of objects, as can be seen below. Thus, there was indeed a need for cluster membership update.
<pre>
- - - c - b - a - - - - - - - x - y - z - - -
     -4  -3  -2               2   3   4      
</pre>

**Solution**: in this implementation, the defintion of *UpdSeed<sub>Del</sub>* is extended to cover such cases. It is (informally) the set of core objects in the *Eps*-neighborhood of either (1) those objects that lose their core object property as a result of the deletion of *p* or (2) *p* itself.

## Updates needed when *UpdSeed<sub>Del</sub>* is empty

According to *Section 4.3* of the paper, when during the deletion of an object *p* if _"UpdSeed<sub>Del</sub> is empty [...] then p is deleted from D and eventually other objects in N<sub>Eps</sub>(p) change from a former cluster C to noise"._

However, consider there are two core objects in *D*, *p* and *q*, not in the *Eps*-neighborhood of each other. They are of different clusters, *C1* and *C2*, respectively. And suppose there is an object *b* that is not core and is in both *N<sub>Eps</sub>(p)* and *N<sub>Eps</sub>(q)* (but not in *N<sub>Eps</sub>(r)* for of any other object *r*). In such cases *b* is either in cluster *C1* or *C2*. In this example assume it is in *C1*.

We now delete *p* from *D*. *UpdSeed<sub>Del</sub>* is empty because there are no core objects in the *Eps*-neighborhood of objects that lost their core property. *b* is then no longer in *C1* (as there is no object to keep it there) but does not become noise. Instead, because it is in *N<sub>Eps</sub>(q)* it should be assigned to *C2*, which goes against the description in the paper.

**Solution**: in this implementation whenever an object loses its cluster membership it is checked first if it should be reassigned to another cluster. Only if it is not in the *Eps*-neighborhood of any other core objects it becomes noise.

## Simultaneous splits

When the paper, in *Section 4.3* (_"potential Split"_), describes the splitting logic that happens after an object *p* is deleted, it says this is when *UpdSeed<sub>Del</sub>* is not empty and the objects in it _"belonged to exactly one cluster [...] before the deletion of p."_

Take the following two dimensional object set *D* as example. There are several objects, most of them marked with a star, and 3 of them with a letter: *p*, *b*, and *q*. With the left and bottom axes one can see the coordinates of the objects.

<pre>
 1   *  *  *     q     *  *  *

 0               b

-1   *  *  *     p     *  *  *
    -2    -1     0     1     2
</pre>

When we cluster these objects according to DBSCAN with *MinPts*=4 and *Eps*=1, two clusters emerge. The first cluster consists of the objects on the *y*=-1 line, while the second one with objects on the line *y*=1. Object *b*, since there are less than *MinPts* objects in *N<sub>Eps</sub>(b)*, is not a core object itself, but it belongs to either one of the clusters as a border object.

What happens when we delete *b* from *D*? As a result, both *q* and *p* lose their core property. According to the definition of *UpdSeed<sub>Del</sub>*, the core objects in the neighborhood of *p* and *q*, that is, the objects marked with stars next to them, will be in *UpdSeed<sub>Del</sub>*. These objects belonged to two clusters, not _"exactly one cluster [...] before the deletion of p"_, as the paper states. The paper misses a point here.

**Solution**: In this case, this implementation follows the logic of DBSCAN and reaches the conclusion of what would happen if DBSCAN was applied to *D* after the deletion of *p*. That is, four clusters are formed, two at the top and two at the bottom. So two splits need happen at the same time: both of the bottom and the top cluster break down into two smaller clusters.
