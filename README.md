![E.ON EBC RWTH Aachen University](./resources/EBC_Logo.png)

# Optimization model of EHDO

This repository contains the **calculation code** of the webtool **EHDO**.
EHDO can be accessed here: https://ehdo.eonerc.rwth-aachen.de

EHDO (Energy hub design optimization) is a user-friendly, free webtool for planning complex multi-energy systems. It is used in academic teaching in **Energy Engineering** at **RWTH Aachen University**. 

## Files

This repository contains 5 files:

- ```optim_model.py```: 
     > The EHDO optimization model with all constraints, variables and objective functions.
- ```load_params.py```: 
     > All model parameters are defined here. User parameters are handed over by the Django webframework.
- ```k_medoids.py```:
     > Preprocessing of the design day clustering algorithm.
- ```clustering_medoid.py```:
	> Optimization model with all constraints and objective functions for the design day clustering.

   
## Publications

The EHDO webtool has been presented in the following publications:

<i>EHDO: A free and open-source webtool for designing and optimizing multi-energy systems based on MILP.</i> M. Wirtz, P. Remmen, D. Müller. Computer Applications in Engineering Education, 2020. DOI: 10.1002/cae.22352. https://onlinelibrary.wiley.com/doi/10.1002/cae.22352 [Open Access]


## License & copyright

Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University

Licensed under the [MIT License](LICENSE).
