IncrementalDBSCAN
-----------------

TODO: Introdoction.

TODO: Animation.


# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#Performance)
- [Notes on the IncrementalDBSCAN paper](#notes-on-the-incrementaldbscan-paper)

# Installation

TODO

# Usage

TODO: API example.

TODO: Notebook examples.

# Performance

TODO

# Notes on the IncrementalDBSCAN paper
The work by Ester et al. 1998 lays the groundwork for this implementation of IncrementalDBSCAN. However, some parts of the algorithm are not covered in the paper. In this section, these holes will be identified, and solutions are proposed to fill them.

Notations used:
- _N<sub>Eps</sub>(p)_: the set of all objects that are in the _Eps_-neighborhood of _p_.
- _UpdSeed<sub>Ins</sub>_, the set of update seeds after insertion, is defined in _Definition 7_ as the set of core objects in the _Eps_-neighborhood of those objects that gain their core object property as a result of the insertion.
- _UpdSeed<sub>Del</sub>_, the set of update seeds after deletion, is defined in _Definition 7_ as the set of core objects in the _Eps_-neighborhood of those objects that lose their core object property as a result of the deletion.

## Absorption when _UpdSeed<sub>Ins</sub>_ is empty
Let's suppose that cluster _C_ is already established, and a new object _p_ is inserted in the _Eps_-neighborhood of a core object _c_ of cluster _C_. Additionally, suppose that there are not enough objects in _N<sub>Eps</sub>(p)_ for _p_ to become a core object and that no other objects become core objects due to the insertion.

Now, how should _p_ be handled?

Since there are no new core objects after the insertion, _UpdSeed<sub>Ins</sub>_ is empty. According to _Section 4.2_, _"if _UpdSeed<sub>Ins</sub>_ is empty [...] then _p_ is a noise object."_ But we also know that _p_ is in _N<sub>Eps</sub>(c)_, and according to _Definition 4_ this means that it should be assigned to cluster _C_. There is clearly a contradiction here.

**Solution**: In this implementation, even if _UpdSeed<sub>Ins</sub>_ is empty, _p_ is assigned to cluster _C_ if _c_ is a core object of cluster _C_ and _p_ is in _N<sub>Eps</sub>(c)_.

## Simultaneous creations, absorptions and merges
In _Section 4.2_, cases of creation, absorption and merge are presented. These are indeed essential building blocks of IncrementalDBSCAN. However, the paper fails to mention that these events can happen simultaneously.

Let's see an example. Suppose we have a 1 dimensional data set with 6 objects (_a_, _b_, _c_, _x_, _y_, _z_) as illustrated in the following block. The coordinates of the objects are as noted below their names.
<pre>
- - - c - b - a - - - - - - - x - y - z - - -
     -4  -3  -2               2   3   4      
</pre>

If we apply IncrementalDBSCAN to the data set with _Eps_=2 and _MinPts_=4, no clusters are created since none of the objects have an _Eps_-neighborhood that contain at least 4 objects. That is, all objects are noise objects.

We now insert object _p_ at position 0.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

After the insertion, both _N<sub>Eps</sub>(a)_ and _N<sub>Eps</sub>(x)_ contain 4 objects, so _a_ and _x_ become core objects. _UpdSeed<sub>Ins</sub>_ then contains the new core objects, _a_ and _x_. According to the paper if _"UpdSeed<sub>Ins</sub> contains only core objects which did not belong to a cluster before the insertion of p, i.e. they were noise objects or equal to p, [...] a new cluster containing these noise objects as well as p is created."_ 

Here _UpdSeed<sub>Ins</sub>_ contains only new core objects (_a_ and _x_) but all 7 of the objects cannot be part of one cluster, since not all objects would be density-reachable from any other object in the cluster (because, e.g., _a_ is not directy density-reachable from _p_). Thus, the definition of a cluster (_Definition 4_) wouldn't hold. This is contradictory to the above quote from _Section 4.2_.

Analogous examples can be constructed for absorptions and merges. E.g., a creation and an absorption can happen at the same time, or even two merges can. But the paper doesn't cover these cases.

**Solution**: _UpdSeed<sub>Ins</sub>_ should be broken down to components in which each object is density-connected to any other object in the component. The rules of creation, absorption and merge should be applied not to _UpdSeed<sub>Ins</sub>_ as a whole but to each component individually.

## Extended definition of _UpdSeed<sub>Del</sub>_

The point of defining _UpdSeed<sub>Del</sub>_ is to take the first step towards finding all objects in the whole object set that eventually might be affected by a deletion. _UpdSeed<sub>Del</sub>_ contains the _"seed objects for the update"_.

Let's take the following object set _D_ of 7 one dimensional objects (_a_, _b_, _c_, _p_, _x_, _y_, _z_). The coordinates of the objects are as noted below their names.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

Suppose we apply IncrementalDBSCAN to the objects with _MinPts_ = 3 and _Eps_ = 2. As a result, all objects belong to a single cluster.

Now suppose we delete _p_. Following _Definition 7_ in the paper, _UpdSeed<sub>Del</sub>_ would be empty, since there is no object that is core in _D_ but not in _D_ \ {_p_}. Thus, according to the definition, there are no seed objects for the update.

This is in conflict with the results of the deletion, in which there are now two clusters of objects, as can be seen below. Thus, there was indeed a need for cluster membership update.
<pre>
- - - c - b - a - - - - - - - x - y - z - - -
     -4  -3  -2               2   3   4      
</pre>

**Solution**: in this implementation, the defintion of _UpdSeed<sub>Del</sub>_ is extended to cover such cases. It is (informally) the set of core objects in the _Eps_-neighborhood of either (1) those objects that lose their core object property as a result of the deletion of _p_ or (2) _p_ itself.

## Updates needed when _UpdSeed<sub>Del</sub>_ is empty

According to _Section 4.3_ of the paper, when during the deletion of an object _p_ if *"_UpdSeed<sub>Del</sub>_ is empty [...] then p is deleted from D and eventually other objects in _N<sub>Eps</sub>(p)_ change from a former cluster C to noise".*

However, consider there are two core objects in _D_, _p_ and _q_, not in the _Eps_-neighborhood of each other. They are of different clusters, _C1_ and _C2_, respectively. And suppose there is an object _b_ that is not core and is in both _N<sub>Eps</sub>(p)_ and _N<sub>Eps</sub>(q)_ (but not in _N<sub>Eps</sub>(r)_ for of any other object _r_). In such cases _b_ is either in cluster _C1_ or _C2_. In this example assume it is in _C1_.

We now delete _p_ from _D_. _UpdSeed<sub>Del</sub>_ is empty because there are no core objects in the _Eps_-neighborhood of objects that lost their core property. _b_ is then no longer in _C1_ (as there is no object to keep it there) but does not become noise. Instead, because it is in _N<sub>Eps</sub>(q)_ it should be assigned to _C2_, which goes against the description in the paper.

**Solution**: in this implementation whenever an object loses its cluster membership it is checked first if it should be reassigned to another cluster. Only if it is not in the _Eps_-neighborhood of any other core objects it becomes noise.

## Simultaneous splits

When the paper, in _Section 4.3_ (_"potential Split"_), describes the splitting logic that happens after an object _p_ is deleted, it says this is when _UpdSeed<sub>Del</sub>_ is not empty and the objects in it "_belonged to exactly one cluster [...] before the deletion of p_".

Take the following two dimensional object set _D_ as example. There are several objects, most of them marked with a star, and 3 of them with a letter: _p_, _b_, and _q_. With the left and bottom axes one can see the coordinates of the objects.

<pre>
 1   *  *  *     q     *   *   *

 0               b

-1   *  *  *     p     *   *   *
    -2    -1     0     1       2
</pre>

When we cluster these objects according to DBSCAN with _MinPts_ = 4 and _Eps_ = 1, two clusters emerge. The first cluster consists of the objects on the _y_=-1 line, while the second one with objects on the line _y_=1. Object _b_, since there are less than _MinPts_ objects in _N<sub>Eps</sub>(b)_, is not a core object itself, but it belongs to either one of the clusters as a border object.

What happens when we delete _b_ from _D_? As a result, both _q_ and _p_ lose their core property. According to the definition of _UpdSeed<sub>Del</sub>_, the core objects in the neighborhood of _p_ and _q_, that is, the objects marked with stars next to them, will be in _UpdSeed<sub>Del</sub>_. These objects belonged to two clusters, not "_exactly one cluster [...] before the deletion of p_", as the paper states. The paper misses a point here.

**Solution**: In this case, this implementation follows the logic of DBSCAN and reaches the conclusion of what would happen if DBSCAN was applied to _D_ after the deletion of _p_. That is, four clusters are formed, two at the top and two at the bottom. So two splits need happen at the same time: both the bottom and the top cluster breaks down into two smaller clusters.

# References
Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Xu, Xiaowei (1996). _A density-based algorithm for discovering clusters in large spatial databases with noise._ In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96). [ACM Digital Library][acm1]. [PDF][pdf1].

Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Wimmer, Michael; Xu, Xiaowei (1998). _Incremental Clustering for Mining in a Data Warehousing Environment._ In: Proceedings of the 24rd International Conference on Very Large Data Bases (VLDB 1998). [ACM Digital Library][acm2]. [PDF][pdf2].

[acm1]: https://dl.acm.org/citation.cfm?id=3001507
[acm2]: https://dl.acm.org/citation.cfm?id=671201
[pdf1]: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
[pdf2]: https://www.dbs.ifi.lmu.de/Publikationen/Papers/VLDB-98-IncDBSCAN.pdf
