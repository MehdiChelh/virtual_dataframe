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


def test_ctr():
    assert numpy.array_equal(
        vnp.asnumpy(vnp.array([1, 2])),
        numpy.array([1, 2])), "can not call compute()"


def test_slicing():
    assert vnp.array([1, 2])[1:].compute(), "can not call compute()"


def test_asnumpy():
    assert isinstance(vnp.asnumpy(vnp.array([1, 2])), numpy.ndarray)

# %% chunks
def test_compute_chunk_sizes():
    vnp.arange(100_000, chunks=(100,)).compute_chunk_sizes().compute()

def test_rechunk():
    vnp.arange(100_000, chunks=(100, )).rechunk((200,200)).compute()

# %% Random
def test_random_random():
    x = vnp.random.random((10000, 10000), chunks=(1000, 1000))
    # FIXME

def test_random_binomial():
    vnp.random.binomial(10,.5,1000,chunks=10)

def test_random_normal():
    vnp.random.normal(0,.1,1000,chunks=10)

def test_random_poisson():
    vnp.random.poisson(5, 10000,chunks=100)

# %% from_...
def test_from_array():
    data = np.arange(100_000).reshape(200, 500)
    a = vnp.from_array(data, chunks=(100, 100))

def test_from_delayed():
    pass

def test_from_npy_stack():
    pass

def test_from_zarr():
    pass
