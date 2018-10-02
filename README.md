# Evolutionary Algorithms for MAXSAT Problems
**Sean Cork, Kamaal Palmer, Luca Ostertag-Hill**

**Nature Inspired Computation: Project 1**

**September 30, 2018**

This evAlt.cpp file contains our Genetic and PBIL algorithm. To run the file:

1. Type `make` to create the executable.
2. To run the Genetic Algorithm type `./evAlg filename pop_size selection crossover cross_prob mut_prob iterations g`
    1. For example `./evAlg t3pm3-5555.spn.cnf 100 rs 1c 0.7 0.01 100 g`
3. To run the PBIL Algorithm type `./evAlg filename pop_size pos_rate neg_rate mut_prob mut_rate iterations p`
    1. For example `./evAlg t3pm3-5555.spn.cnf 100 0.1 0.075 0.02 0.05 1000 p`
