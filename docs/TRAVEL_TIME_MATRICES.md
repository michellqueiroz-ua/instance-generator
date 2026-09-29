# Travel-time matrices and the triangle inequality

REQreate's generated travel-time matrices are built to satisfy the directed
triangle inequality: travelling directly from node A to node C is no slower
than travelling from A to C via another node B. The underlying shortest-path
values satisfy this property. Persisted matrix entries are integer seconds, so
sub-second truncation can produce a discrepancy of at most one second.

The matrix must be built from shortest paths weighted by each edge's
`travel_time`. Road speeds differ, so the shortest-distance path is not
necessarily the shortest-time path. Converting the length of a shortest-distance
path with a fixed speed can therefore violate the travel-time triangle
inequality.

REQreate computes the drive matrix with `weight="travel_time"` and uses that
precomputed matrix for drive-time estimates. Thus generated instances preserve
the triangle inequality up to that one-second rounding tolerance. Cached
networks created before this behavior was introduced must be regenerated to
receive the guarantee.
