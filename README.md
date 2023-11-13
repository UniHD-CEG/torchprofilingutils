# Torch Profiling Utils
This module contains two classes that simplify profiling Torch models using the [fvcore](https://github.com/facebookresearch/fvcore/tree/main) and [torchinfo](https://github.com/TylerYep/torchinfo) libraries.
I designed the wrapper classes `FVCoreWriter` and `TorchinfoWriter` contained in this module around my profiling workflow and am sharing them in hopes that they may be useful for other group members.

## Installation
To install the module, run `make install` in the project root folder.

## Usage

### FVCoreWriter
The `FVCoreWriter` class wraps the Facebook Research [fvcore](https://github.com/facebookresearch/fvcore/tree/main) module.
It allows the retrieval of the FLOPs and activation counts of a model as a `dict`.
It also contains functions to store the retrieved values as JSON files.
The class is instantiated with the model to be profiled as well as the input data the model receives in the forward pass. If the model has multiple input parameters, pass them as as `tuple`.

```python
from torch_profiling_utils.fvcorewriter import FVCoreWriter

fvcore_writer = FVCoreWriter(model, input_data)
```

After instantiation, the FLOPs and activation counts, either by module or by operator, can be retrieved as `dict`s in the following manner:

```python
fvcore_writer.get_flop_dict('by_module')
fvcore_writer.get_flop_dict('by_operator')

fvcore_writer.get_activation_dict('by_module')
fvcore_writer.get_activation_dict('by_operator')
```

The FLOPs and activation counts can also be directly stored in JSON format using

```python
fvcore_writer.write_flops_to_json(output_filename_string, 'by_module')
fvcore_writer.write_flops_to_json(output_filename_string,'by_operator')

fvcore_writer.write_activations_to_json(output_filename_string, 'by_module')
fvcore_writer.write_activations_to_json(output_filename_string,'by_operator')
```

### TorchinfoWriter

```python
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()
```

```python
torchinfo_writer.show_model_tree(attr_list=['Parameters', 'MACs'])
```

```python
torchinfo_writer.get_dataframe()
```

```python
torchinfo_writer.get_dot()

```



## Tested Environment
This module was tested using the following package/Python module versions:
| Package/Module | Version |
| ---            | ---     |
| python         | 3.9.12  |
| numpy          | 1.21.2  |
| pandas         | 1.4.1   |
| pytorch        | 1.11.0  |
| fvcore         | 0.1.5   |
| torchinfo      | 1.8.0   |
| bigtree        | 0.14.4  |

