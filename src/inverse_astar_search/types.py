from typing import List, Mapping, Tuple

Node = int

NodePair = Tuple[Node, Node]

Edge = NodePair

Path = List[Node]

Paths = List[Path]

NodeIndexedPaths = Mapping[Node, Paths]

TripIndexedPaths = Mapping[NodePair, Paths]

NodePairIndex = Mapping[NodePair, int]


