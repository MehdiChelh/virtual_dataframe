import numpy as np
import pandas

import virtual_dataframe as vdf


def test_array():
    assert vdf.numpy.array_equal(
        vdf.numpy.asnumpy(vdf.numpy.array([
        [0.0, 1.0, 2.0, 3.0],
        [1, 2, 3, 4],
        [10, 20, 30, 40]
        ])),
        np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1, 2, 3, 4],
            [10, 20, 30, 40]
        ])
    )


def test_DataFrame_to_narray():
    df = vdf.VDataFrame(
        {'a': [0.0, 1.0, 2.0, 3.0],
         'b': [1, 2, 3, 4],
         'c': [10, 20, 30, 40]
         },
        npartitions=2
    )
    a = df.to_narray()
    assert vdf.numpy.array_equal(
        vdf.numpy.asnumpy(a),
        np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1, 2, 3, 4],
            [10, 20, 30, 40]
        ]).T)

def test_Series_to_narray():
    serie = vdf.VSeries(
        [0.0, 1.0, 2.0, 3.0],
        npartitions=2
    )
    a = serie.to_narray()
    assert vdf.numpy.array_equal(
        vdf.numpy.asnumpy(a),
        np.array(
            [0.0, 1.0, 2.0, 3.0],
        ))
