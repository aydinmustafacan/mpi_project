# Parallel Programming using MPI Interface

This project’s aim is to simulate parallel programming interface for relief algorithm. In this project I implement an algorithm called relief with MPI interface. With the help of MPI interface I will be simulating multiple processor environment working parallel to solve relief algorithm as fast as possible.

# Running

~~~~~~~~~~~~~~~{.cpp}
   mpic++ -o executable ./main.cpp
    
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~{.cpp}
   mpirun --oversubscribe -np Number executable ./test.tsv
~~~~~~~~~~~~~~~



