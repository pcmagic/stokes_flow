# array module example
import sample
import array

a = array.array('d', [1, -3, 4, 7, 2, 0])
PETSc.Sys.Print(a)
sample.clip(a, 1, 4, a)
PETSc.Sys.Print(a)

# numpy example
import numpy

b = numpy.random.uniform(-10, 10, size=1000000)
PETSc.Sys.Print(b)
c = numpy.zeros_like(b)
PETSc.Sys.Print(c)
sample.clip(b, -5, 5, c)
PETSc.Sys.Print(c)
PETSc.Sys.Print(min(c))
PETSc.Sys.Print(max(c))

# Timing test
from timeit import timeit

PETSc.Sys.Print('numpy.clip')
PETSc.Sys.Print(timeit('numpy.clip(b,-5,5,c)', 'from __main__ import b,c,numpy', number=1000))
PETSc.Sys.Print('sample.clip')
PETSc.Sys.Print(timeit('sample.clip(b,-5,5,c)', 'from __main__ import b,c,sample', number=1000))

PETSc.Sys.Print('sample.clip_fast')
PETSc.Sys.Print(timeit('sample.clip_fast(b,-5,5,c)', 'from __main__ import b,c,sample', number=1000))

# 2D test
d = numpy.random.uniform(-10, 10, size=(1000, 1000))
PETSc.Sys.Print(d)
sample.clip2d(d, -5, 5, d)
PETSc.Sys.Print(d)
