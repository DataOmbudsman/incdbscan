IncrementalDBSCAN
-----------------

TODO: Introdoction.

TODO: Animation.

# Table of contents

TODO

# Usage

TODO: API example.

TODO: Notebook examples.

# Complexity

TODO

# Additions to Ester et al. 1998
The work by Ester et al. 1998 lays the groundwork for the incremental version of DBSCAN. However, two edge cases are not covered in the paper. In this section, these holes will be identified, and solutions are proposed to fill them.

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

We now add point _p_ to position 0.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

After the insertion, both _N<sub>Eps</sub>(a)_ and _N<sub>Eps</sub>(x)_ contain 4 objects, so _a_ and _x_ become core objects. _UpdSeed<sub>Ins</sub>_ then contains the new core objects, _a_ and _x_. According to the paper if _"UpdSeed<sub>Ins</sub> contains only core objects which did not belong to a cluster before the insertion of p, i.e. they were noise objects or equal to p, [...] a new cluster containing these noise objects as well as p is created."_ 

Here _UpdSeed<sub>Ins</sub>_ contains only new core objects (_a_ and _x_) but all 7 of the objects cannot be part of one cluster, since not all objects would be density-reachable from any other object in the cluster (because, e.g., _a_ is not directy density-reachable from _p_). Thus, the definition of a cluster (_Definition 4_) wouldn't hold. This is contradictory to the above quote from _Section 4.2_.

Analogous examples can be constructed for absorptions and merges. E.g., a creation and an absorption can happen at the same time, or even two merges can. But the paper doesn't cover these cases.

**Solution**: _UpdSeed<sub>Ins</sub>_ should be broken down to components in which each object is density-connected to any other object in the component. The rules of creation, absorption and merge should be applied not to _UpdSeed<sub>Ins</sub>_ as a whole but to each component individually.

## Extended definition of _UpdSeed<sub>Del</sub>_

The point of defining _UpdSeed<sub>Del</sub>_ is the first step towards finding all objects in the whole object set that eventually might be affected by a deletion. _UpdSeed<sub>Del</sub>_ contains the _"seed objects for the update"_.

Let's take the following object set _D_ of 7 one dimensional objects (_a_, _b_, _c_, _p_, _x_, _y_, _z_). The coordinates of the objects are as noted below their names.
<pre>
- - - c - b - a - - - p - - - x - y - z - - -
     -4  -3  -2       0       2   3   4      
</pre>

Suppose we apply IncrementalDBSCAN to the objects with _MinPts_ = 3 and _Eps_ = 2. As a result, all objects belong to a single cluster.

Now suppose we remove _p_. Following _Definition 7_ in the paper, _UpdSeed<sub>Del</sub>_ would be empty, since there is no object that is core in _D_ but not in _D_ \ {_p_}. Thus, according to the definition, there are no seed objects for the update.

This is in conflict with the results of the deletion, in which there are now two clusters of objects, as can be seen below. Thus, there was indeed a need for cluster membership update.
<pre>
- - - c - b - a - - - - - - - x - y - z - - -
     -4  -3  -2               2   3   4      
</pre>

**Solution**: in this implementation, the defintion of _UpdSeed<sub>Del</sub>_ is extended to cover such cases. It is (informally) the set of core objects in the _Eps_-neighborhood of either (1) those objects that lose their core object property as a result of the deletion of _p_ or (2) _p_ itself.

# References
Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Xu, Xiaowei (1996). _A density-based algorithm for discovering clusters in large spatial databases with noise._ In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96). [ACM Digital Library][acm1]. [PDF][pdf1].

Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Wimmer, Michael; Xu, Xiaowei (1998). _Incremental Clustering for Mining in a Data Warehousing Environment._ In: Proceedings of the 24rd International Conference on Very Large Data Bases (VLDB 1998). [ACM Digital Library][acm2]. [PDF][pdf2].

[acm1]: https://dl.acm.org/citation.cfm?id=3001507
[acm2]: https://dl.acm.org/citation.cfm?id=671201
[pdf1]: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
[pdf2]: https://www.dbs.ifi.lmu.de/Publikationen/Papers/VLDB-98-IncDBSCAN.pdf
