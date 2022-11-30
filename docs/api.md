## API

| api                                    | comments                                                                            |
|----------------------------------------|-------------------------------------------------------------------------------------|
| VDF_MODE                               | The current mode                                                                    |
| Mode                                   | The enumeration of differents mode                                                  |
| FrontEnd                               | `pandas`, `cudf`, `modin.pandas`, `dask_cudf`, `dask.dataframe` or `pyspark.pandas` |
| FrontEndNumpy                          | `numpy`, `cupy` or `dask.array`                                                     |
| BackEndDataFrame                       | `pandas` or `cudf`                                                                  |
| BackEndSeries                          | `pandas.Series`, `cudf.Series` or `modin.pandas.Series`                             |
| BackEndNDArray                         | `numpy.ndarray` or `cupy.ndarray`                                                   |
| BackEnd                                | `pandas`, `cudf` or `modin.pandas`                                                  |
| vdf.@delayed                           | Delayed function (do nothing or dask.delayed)                                       |
| vdf.concat(...)                        | Merge VDataFrame                                                                    |
| vdf.read_csv(...)                      | Read VDataFrame from CSVs *glob* files                                              |
| vdf.read_excel(...)<sup>*</sup>        | Read VDataFrame from Excel *glob* files                                             |
| vdf.read_fwf(...)<sup>*</sup>          | Read VDataFrame from Fwf *glob* files                                               |
| vdf.read_hdf(...)<sup>*</sup>          | Read VDataFrame from HDFs *glob* files                                              |
| vdf.read_json(...)                     | Read VDataFrame from Jsons *glob* files                                             |
| vdf.read_orc(...)                      | Read VDataFrame from ORCs *glob* files                                              |
| vdf.read_parquet(...)                  | Read VDataFrame from Parquets *glob* files                                          |
| vdf.read_sql_table(...)<sup>*</sup>    | Read VDataFrame from SQL                                                            |
| vdf.from_pandas(pdf, npartitions=...)  | Create Virtual Dataframe from Pandas DataFrame                                      |
| vdf.from_backend(vdf, npartitions=...) | Create Virtual Dataframe from backend dataframe                                     |
| vdf.compute([...])                     | Compute multiple @delayed functions                                                 |
| VDataFrame(data, npartitions=...)      | Create DataFrame in memory (only for test)                                          |
| VSeries(data, npartitions=...)         | Create Series in memory (only for test)                                             |
| VLocalCluster(...)                     | Create a dask Local Cluster (Dask, Cuda or Spark)                                   |
| BackEndDataFrame                       | The class of dask backend dataframe                                                 |
| BackEndSeries                          | The class of dask backend series                                                    |
| BackEnd                                | The backend framework                                                               |
| VDataFrame.compute()                   | Compute the virtual dataframe                                                       |
| VDataFrame.persist()                   | Persist the dataframe in memory                                                     |
| VDataFrame.repartition()               | Rebalance the dataframe                                                             |
| VDataFrame.visualize()                 | Create an image with the graph                                                      |
| VDataFrame.to_pandas()                 | Convert to pandas dataframe                                                         |
| VDataFrame.to_csv()                    | Save to *glob* files                                                                |
| VDataFrame.to_excel()<sup>*</sup>      | Save to *glob* files                                                                |
| VDataFrame.to_feather()<sup>*</sup>    | Save to *glob* files                                                                |
| VDataFrame.to_hdf()<sup>*</sup>        | Save to *glob* files                                                                |
| VDataFrame.to_json()                   | Save to *glob* files                                                                |
| VDataFrame.to_orc()                    | Save to *glob* files                                                                |
| VDataFrame.to_parquet()                | Save to *glob* files                                                                |
| VDataFrame.to_sql()<sup>*</sup>        | Save to sql table                                                                   |
| VDataFrame.to_numpy()                  | Convert to numpy array                                                              |
| VDataFrame.to_backend()                | Convert to backend dataframe                                                        |
| VDataFrame.categorize()                | Detect all categories                                                               |
| VDataFrame.apply_rows()                | Apply rows, GPU template                                                            |
| VDataFrame.map_partitions()            | Apply function for each parttions                                                   |
| VDataFrame.to_ndarray()                | Convert to numpy or cupy ndarray                                                    |
| VDataFrame.to_numpy()                  | Convert to numpy, cupy or dask array                                                |
| VSeries.compute()                      | Compute the virtual series                                                          |
| VSeries.persist()                      | Persist the dataframe in memory                                                     |
| VSeries.repartition()                  | Rebalance the dataframe                                                             |
| VSeries.visualize()                    | Create an image with the graph                                                      |
| VSeries.to_pandas()                    | Convert to pandas series                                                            |
| VSeries.to_backend()                   | Convert to backend series (pandas, cudf, dask.Series)                               |
| VSeries.to_numpy()                     | Convert to numpy ndarray                                                            |
| VSeries.to_ndarray()                   | Convert to numpy backend (numpy, cupy or dask.array)                                |
| VClient(...)                           | The connexion with the cluster                                                      |
| import vdf.numpy                       | Import numpy, cupy or dask.array                                                    |
| vdf.numpy.array(..., chunks=...)       | Create an numpy-like array                                                          |
| vdf.numpy.arange(...)                  | Return evenly spaced values within a given interval.                                |
| vdf.numpy.zeros(...)                   | Return a new array of given shape and type, filled with zeros.                      |
| vdf.numpy.zeros_like(...)              | Return an array of zeros with the same shape and type as a given array.                      |
| vdf.numpy.asnumpy(d)                   | Convert to `numpy.ndarray`                                                          |
| vdf.numpy.save(d)                      | Save to npy format                                                                  |
| vdf.numpy.load(d)                      | Load npy format                                                                     |
| vdf.numpy.random.xxx(..., chunks=...)  | Generate random values                                                              |

<sup>*</sup> some frameworks do not implement it


You can read a sample notebook [here](https://github.com/pprados/virtual-dataframe/blob/master/notebooks/demo.ipynb)
for an exemple of all API.

Each API propose a specific version for each framework. For example:

- the  `toPandas()` with Panda, return `self`
- `@delayed` use the dask `@delayed` or do nothing, and apply the code when the function was called.
In the first case, the function return a part of the dask graph. In the second case, the function return immediately
the result.
- `read_csv("file*.csv")` read each file in parallele with dask, spark or pyspark by each worker,
or read sequencially each file and combine each dataframe with a framework without distribution (pandas, modin, cudf)
- `save_csv("file*.csv")` write each file in parallele with dask, spark or pyspark,
or write one file `"file.csv"` (without the star) with a framework without distribution (pandas, modin, cudf)
- ...

All adjustement was describe [here](tech.md)

