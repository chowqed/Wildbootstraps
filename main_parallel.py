# main program that run the simulations

from generateData import *
import sys
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


i = int(sys.argv[1])
def coverageProb(i):
	k=0
	for num in range(0,i):
		test = getIVEst(genData(n=1000,delta=0.5))
		k = getAsyCI(1000,test, 1.64)+k
	return k


local_n = i/size
count = np.zeros(1)
total = np.zeros(1)

count[0] = coverageProb(local_n)

comm.Reduce(count, total, op=MPI.SUM, root=0)


if comm.rank == 0:
        print "With n =", i, "the coverage probability is ", total/i
