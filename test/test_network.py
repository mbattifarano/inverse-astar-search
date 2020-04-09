import numpy as np

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays

square = (integers(min_value=5, max_value=20)
           .map(lambda n: (n, n))
          )

weighted_adjacency_matrix = arrays(
    dtype=np.float,
    shape=square,
    elements=integers(min_value=0, max_value=10)
)

@given(A=weighted_adjacency_matrix)
def test_random_networks(network_factory, A):
    graph = network_factory(A)
    assert graph.number_of_nodes() >= 5
