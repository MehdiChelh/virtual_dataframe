from functools import update_wrapper
import numpy
from inspect import getmembers
import sys

FrontEndNumpy = numpy

# Inject all numpy api here
_module = sys.modules[__name__]  # Me
for k, v in getmembers(numpy, None):
    setattr(_module, k, getattr(numpy, k))


# Wrapper to inject some methods
class Vnarray(numpy.ndarray):

    def compute(self):
        return self

    def __new__(subtype, shape,  # copy=True, order='K', subok=False, ndmin=0, like=None
                dtype=None,
                buffer=None,
                offset=0,
                strides=None,
                order=None):
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        # return Vnarray._add_methods(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Vnarray._add_methods(self)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # - *ufunc* is the ufunc object that was called.
        # - *method* is a string indicating how the Ufunc was called, either
        #   ``"__call__"`` to indicate it was called directly, or one of its
        #   :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
        #   ``"reduceat"``, ``"outer"``, or ``"at"``.
        # - *inputs* is a tuple of the input arguments to the ``ufunc``
        # - *kwargs* contains any optional or keyword arguments passed to the
        #   function. This includes any ``out`` arguments, which are always
        #   contained in a tuple.
        print(f"call __array_ufunc__({ufunc=},{method=},{inputs=},{out=},{kwargs=})")
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Vnarray):
                in_no.append(i)
                args.append(input_.view(numpy.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Vnarray):
                    out_no.append(j)
                    out_args.append(output.view(numpy.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        # info = {}
        # if in_no:
        #     info['inputs'] = in_no
        # if out_no:
        #     info['outputs'] = out_no
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            # if isinstance(inputs[0], Vnarray):
            # inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((numpy.asarray(result).view(Vnarray)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        # if results and isinstance(results[0], Vnarray):
        #     Vnarray._add_methods(results[0])

        return results[0] if len(results) == 1 else results


def array(*args, **kwds):
    return numpy.array(*args, **kwds).view(Vnarray)


update_wrapper(array, numpy.array)
