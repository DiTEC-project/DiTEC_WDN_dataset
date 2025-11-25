 # DiTEC-WDN

This is the official repository for the paper:  
**DiTEC-WDN: A Large-Scale Dataset of Hydraulic Scenarios across Multiple Water Distribution Networks**.

This repository contains configuration optimization, scenario generation, and encapsulation code for the **DiTEC-WDN** dataset.

This is useful for individuals or organizations to generate scenarios on their own private Water Distribution Networks. 

Those interested in the data can directly refer to the [dataset](https://huggingface.co/datasets/rugds/ditec-wdn).

# Tutorial
Access the wiki at [https://ditec-project.github.io/DiTEC_WDN_dataset](https://ditec-project.github.io/DiTEC_WDN_dataset) for more details.


# Repo map
```
    |-- arguments   - where configs stored
    |-- core        - code for interface, demand generator, simgen
    |-- inputs      - contains original INP files
    |-- opt         - code for PSO
    |-- utils       - where we access utils functions
    |-- vis         - code for visualization
    |-- docs        - documentation how to use modules & inteface
```

# License
MIT license. See the LICENSE file for more details.

# Citing DiTEC-WDN

If you use the dataset or the code, please cite:

```latex
@article{truong2025dwd,
  author    = {Huy Truong and Andr{\'e}s Tello and Alexander Lazovik and Victoria Degeler},
  title     = {DiTEC-WDN: A Large-Scale Dataset of Hydraulic Scenarios across Multiple Water Distribution Networks},
  journal   = {Scientific Data},
  year      = {2025},
  volume    = {12},
  number    = {1},
  pages     = {1733},
  doi       = {10.1038/s41597-025-06026-0},
  url       = {https://doi.org/10.1038/s41597-025-06026-0},
  issn      = {2052-4463}
}
```
