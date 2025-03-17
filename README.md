# Biogeography
Modeling of Species

## MicroGP (Model)
The MicroGP model is found in [MicroGP.py](development/MicroGP.py)

## Config
The [config.py](configs/config.py) file contains the different configurations.

### Config Sharing
Should the data be stored in different places and one would like to use the same config file, then please add a variable with the path to the folder containing the data.
Currently, [data_folder_path.py](configs/data_folder_path.py) is containing the variable 'data_folder_path = "/path/to/data"' – should it not be created, please add the variable with your intended path.  
