# POMDP PBVI Exploration

In this repository, we explore the Point-Based Value Iteration (PBVI) algorithm to solve Partially-Observable Markov Decision Processes (POMDP).

The PBVI algorithm comes in different flavors as described in "J Pineau, et al.; Point-based value iteration: An anytime algorithm for POMDPs; 2003". The different flavors of PBVI are:

- Random belief selection (RA)
- Stochastic Search with Random Action (SSRA)
- Stochastic Search with Greedy Action (SSGA)
- Stochastic Search with Exploratory Action (SSEA)
- Greedy Error Reduction (GER)

Then more techniques were described in different papers and summarized in: "G. Shani, et al.; A survey of point-based POMDP solvers; 2012"

- Perseus
- Heuristic Search Value Iteration (HSVI)
- Forward Search Value iteration (FSVI)

These techniques were then tested on 2 and 3-state toy problems to see how the details of the PBVI process. Then it was applied to the problem of the Olfactory Navigation which is a much more complex problem with a large amount of states (to the order of 30k). With this, optimizations have been done to the PBVI algorithm to allow for GPU operations to be performed. Alongside this, we further improved the PBVI algorithm to make use of the determinicity of the olfactory navigation problem (the transition matrix is very sparse). This was done by introducing a notion of reachability (which states are reachable from any other state) to minimize the amount of operations required within the PBVI algorithm and therefore reducing the computation complexity.

# Master Thesis

This work has been performed as part of the Master's Thesis of Arnaud Ruymaekers at the Università di Genova under the supervision of Professor A. Seminara and Professor A. Verri.

# Doc style reference
https://numpydoc.readthedocs.io/en/latest/format.html#parameters
