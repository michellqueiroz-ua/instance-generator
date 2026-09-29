# Travel-time matrices and the triangle inequality

REQreate's generated travel-time matrices satisfy the directed triangle
inequality: travelling directly from node A to node C is no slower than
travelling from A to C via another node B. Routing algorithms can rely on this
property, subject only to normal floating-point precision.

The matrix must be built from shortest paths weighted by each edge's
`travel_time`. Road speeds differ, so the shortest-distance path is not
necessarily the shortest-time path. Converting the length of a shortest-distance
path with a fixed speed can therefore violate the travel-time triangle
inequality.

REQreate computes the drive matrix with `weight="travel_time"` and uses that
precomputed matrix for drive-time estimates. Cached networks created before this
behavior was introduced must be regenerated to receive the guarantee.
