[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# On the Value of Risk-Averse Multistage Stochastic Programming in Capacity Planning

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc)
under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[On the Value of Risk-Averse Multistage Stochastic Programming in Capacity Planning](https://doi.org/10.1287/ijoc.2023.0396) by Xian Yu and Siqian Shen.

# Cite

To cite this software, please cite the paper using its DOI and the software DOI.

https://doi.org/10.1287/ijoc.2023.0396

https://doi.org/10.1287/ijoc.2023.0396.cd

Below is the BibTex for citing this version of the code.

```
@article{yu2024repo,
  author =        {X. Yu and S. Shen},  
  publisher =     {INFORMS Journal on Computing},
  title =         {{On the Value of Risk-Averse Multistage Stochastic Programming in Capacity Planning}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0396.cd},
  url =           {https://github.com/INFORMSJoC/2023.0396},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0396},
}
```

# Description
This repository aims to demonstrate the value of risk-averse multistage stochastic programming models in capacity planning problems. 

We consider a risk-averse stochastic capacity planning problem under uncertain demand in each period. Using a scenario tree representation of the uncertainty, we formulate a multistage stochastic integer program to adjust the capacity expansion plan dynamically as more information on the uncertainty is revealed. We compare it with a two-stage approach that determines the capacity acquisition for all the periods up front.  Using expected conditional risk measures (ECRMs), we derive a tight lower bound and an upper bound for the gaps between the optimal objective values of risk-averse multistage models and their two-stage counterparts. Furthermore, we propose approximation algorithms to solve the two models more efficiently, which are asymptotically optimal under an expanding market assumption. We conduct numerical studies using randomly generated and real-world instances with diverse sizes, to demonstrate the tightness of the analytical bounds and efficacy of the approximation algorithms.

# Data files
```data/data49UFLP.xls``` contains the 49 candidate facility locations from the paper "Daskin MS (2011) Network and Discrete Location: Models, Algorithms, and Applications" <br />
```data/data88UFLP.xls``` contains the 88 customer sites from the paper "Daskin MS (2011) Network and Discrete Location: Models, Algorithms, and Applications" 

# Code files
```scripts/BoundStochasticProgram.py``` contains all the main functions <br />
```computeVMS.py``` computes the VMS, relative VMS, and its lower and upper bounds, using synthetic dataset <br />
```approximateMultistageProgram.py``` approximates the multistage stochastic program using the proposed approximation algorithm and compares it with SDDiP <br />
```caseStudy.py``` computes the optimal solutions and costs of two-stage and multistage models using real-world dataset from "Daskin MS (2011) Network and Discrete Location: Models, Algorithms, and Applications"
