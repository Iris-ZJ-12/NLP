# Utils

## Introduction

This module contains some useful common tools for AI projects.

## Layout
- [sm_util](sm_util.py): Util functions for simpletransformers package.
    - `auto_rm_outputs_dir`: a method used after training to remove `/outputs/` 
    directory which is useless and large.
    - `eval_ner`: evaluate a dataframe for NER tasks.

  
- [api_util](api_util.py): Decorator functions for logging inputs and outputs of a function.
    
    - `Logger`: The logger class.
    
        - Properties:
        
            - `default_format`: String. Default logging format.
        
        - Methods:
        
            - `log_input_output()`: Decorator for functions in `api.py`. 
            `result.log` is written with input and output variables.
        
            
- [sm_util](sm_util.py): utility functions for simpletransformers
    
    - `hide_labels_arg(model)`: The wrapper function of NERmodel.train_model().
    
        - Function: Automatically generate labels from training dataset
        and set them to model args.
    
        - Input: NERmodel
        
        - Output: The function similar to `NERmodel.train_model()`