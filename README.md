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

If you use the dataset, please cite:

```latex
@misc{truong2025dwd,
      title={DiTEC-WDN: A Large-Scale Dataset of Hydraulic Scenarios across Multiple Water Distribution Networks}, 
      author={Huy Truong and Andrés Tello and Alexander Lazovik and Victoria Degeler},
      year={2025},
      eprint={2503.17167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.17167},
      note = {HT and AT contributed equally to this work. The dataset is linked to a paper submitted to *Nature Scientific Data*.}
}
```
