import numpy as np
import numpy

import virtual_dataframe as vdf
import virtual_dataframe.numpy as vnp


def test_array():
    assert numpy.array_equal(
        vnp.asnumpy(vnp.array([
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


def test_DataFrame_to_ndarray():
    df = vdf.VDataFrame(
        {'a': [0.0, 1.0, 2.0, 3.0],
         'b': [1, 2, 3, 4],
         'c': [10, 20, 30, 40]
         },
        npartitions=2
    )
    a = df.to_ndarray()
    assert numpy.array_equal(
        vnp.asnumpy(a),
        np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1, 2, 3, 4],
            [10, 20, 30, 40]
        ]).T)


def test_Series_to_ndarray():
    serie = vdf.VSeries(
        [0.0, 1.0, 2.0, 3.0],
        npartitions=2
    )
    a = serie.to_ndarray()
    assert numpy.array_equal(
        vnp.asnumpy(a),
        np.array(
            [0.0, 1.0, 2.0, 3.0],
        ))


def test_asarray():
    df = vdf.VDataFrame(
        {'a': [0.0, 1.0, 2.0, 3.0],
         'b': [1, 2, 3, 4],
         'c': [10, 20, 30, 40]
         },
        npartitions=2
    )
    a = vnp.asarray(df['a'])
    numpy.array_equal(
        vnp.asnumpy(a),
        np.array(
            [0.0, 1.0, 2.0, 3.0],
        ))


def test_DataFrame_ctr():
    a = vnp.array([
        [0.0, 1.0, 2.0, 3.0],
        [1, 2, 3, 4],
        [10, 20, 30, 40]
    ])
    df1 = vdf.VDataFrame(a.T)
    df2 = vdf.VDataFrame(
        {
            0: [0.0, 1.0, 2.0, 3.0],
            1: [1.0, 2.0, 3.0, 4.0],
            2: [10.0, 20.0, 30.0, 40.0]
        }
    )
    assert vdf.compute((df1 == df2).all().all())


def test_ctr_serie():
    a = vnp.array([0.0, 1.0, 2.0, 3.0])
    vdf.VSeries(a)


def test_random():
    # FIXME x = vnp.random.random((10000, 10000), chunks=(1000, 1000))
    x = vnp.random.random((10000, 10000))
    # FIXME


def test_ctr():
    assert numpy.array_equal(
        vnp.asnumpy(vnp.array([1, 2])),
        numpy.array([1, 2])), "can not call compute()"


def test_slicing():
    assert vnp.array([1, 2])[1:].compute(), "can not call compute()"


def test_asnumpy():
    assert isinstance(vnp.asnumpy(vnp.array([1, 2])), numpy.ndarray)
