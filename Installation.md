# Guide to Run Battle of Networks 2024 (BoN 2024)

## Prerequisites
- Python Version: The models for BoN 2024 were trained using **Python 3.11.6**. Ensure you have this version installed. You can download it from Python 3.11.6 Download (https://www.python.org/downloads/release/python-3116/).
- For Mac Users: Homebrew installed (https://brew.sh)

## Setting Up a Virtual Environment
To manage dependencies effectively, it's recommended to set up a virtual environment (`venv`). This isolates the project dependencies from your global Python installation. Follow these steps:

### Windows (with Visual Studio>2015 installed)
1. Create a Virtual Environment:
  - On Windows open Command Prompt (press `Windows Key + R`, then type `cmd` and press Enter).
  - Create a new virtual environment by running:
    `python -m venv /path/to/new/virtual/environment_name`
    Replace `/path/to/new/virtual/environment_name` with your desired directory path.

2. Activate the Virtual Environment:
  - Activate your virtual environment by running:
    `/path/to/new/virtual/environment_name/Scripts/activate`.
    Ensure you use the correct path where your virtual environment is located.

3. Install Required Modules:
  - Browse to the code directory of the Water-Futures BoN2024 codebase
    `cd /path/to/the/code/BoN2024`
  - Install all necessary modules by running:
    `pip install -r requirements_windows.txt`

### Mac with Apple Silicon
1. Create a Virtual Environment:
  - Open a terminal.
  - Create a new virtual environment by running:
    `python -m venv /path/to/new/virtual/environment_name`
    Replace `/path/to/new/virtual/environment_name` with your desired directory path.

2. Activate the Virtual Environment:
  - Activate your virtual environment by running:
    `source /path/to/new/virtual/environment_name/bin/activate`.
    Ensure you use the correct path where your virtual environment is located.

3. Install OpenMP library
  - `brew install libomp`

4. Install Required Modules:
  - Browse to the code directory of the Water-Futures BoN2024 codebase
    `cd /path/to/the/code/BoN2024`
  - Install all necessary modules by running:
    `pip install -r requirements_mac.txt`

### Manual installation
1. Follow steps 1 and 2 from the previous section based on your system.

2. Install the following libraries using the instructions for your machine on each website:
- pytorch (https://pytorch.org)
- tensorflow (https://www.tensorflow.org/install/pip)
- lightbm (https://pypi.org/project/lightgbm/) -> `pip install lightgbm`
- darts (https://github.com/unit8co/darts/blob/master/INSTALL.md) -> `pip install "u8darts[notorch]"`
- jupyter (https://jupyter.org/install) -> `pip install jupyter`
- plotly (https://plotly.com/python/getting-started/) -> `pip install plotly==5.18.0`
- dash (https://dash.plotly.com/installation) -> `pip install dash`
- openpyxl (https://pypi.org/project/openpyxl/) -> `pip insall openpyxl`
- adtk (https://pypi.org/project/adtk/) -> `pip install adtk`
- metpy (https://pypi.org/project/MetPy/) -> `pip install metpy`
- dm-tree (https://pypi.org/project/dm-tree/) -> `pip install dm-tree`
- torchmetrics (https://pypi.org/project/torchmetrics/) -> `pip install torchmetrics`

## Download the data
If you want to produce new models and results for reproducibility, you can skip this part: the code is already set up to use the *data* directory in this project. 

On the other hand, if you downloaded this repo to investiagte the results produced by our team, follow the following steps:
1. download the data folder from Zenodo (#TBD).
2. change the value of the environmental variable `BON2024_DATA_FOLDER` in the `.env` file present in this repository to the **full path** of the downloaded data folder.

## Running the Code
After setting up the virtual environment and installing the dependencies, you're ready to run the code. 
Run the `water_futures.py` file or using Jupyter Notebook, select the correct Kernel and run `water_futures.ipynb`.

## Additional Notes
### GPU Acceleration
BoN 2024 can leverage NVIDIA GPU for enhanced performance. If your system does not have an NVIDIA GPU, modify the configuration in the 'water_futures.xx' file:

Change Line 265 from:
cfg['device'] = 'cuda'
to:
cfg['device'] = 'cpu'

#### Note: the Metal acceleration on Mac with M1 or M2 is not working at the time of the submission. Use 'cpu' to ensure a correct running on Apple devices, too. 

## Important Note
The virtual environment for this project includes both TensorFlow and PyTorch. Be aware that this can consume significant storage space.

# Results
The forecast results can be found at:
`data/results/strategies/avg_top5/avg_top5__iter_1__eval__.xlsx`
or:
*data/solutions/WaterFutures_SolutionTemplate_W1.xlsx*


# Model Run Times

Each models takes the following times during training, if you aim to run the entire project account for long times (more than two days on a single machine). We have 52 iterations during training, 4 in testing and 1 for the final evaluation. All models are run for multiple seeds (for the solution we used n_train_seed=1 and n_test_seed =3 ), but a more appropriate and complete solution should use 5/10 seed.  

The reported times are from a Mac with an 8 core (4+4) Apple silicon M1 processor and 16 GB of RAM. 

Model				Time(s/iteration)
-----------------------------------
LightGBM Simple		|	285
LightGB Robust | 555
Light GBM last week | 283
XGBM | 281
Wavenet				| 3250 (55 min)

All other non deep-learning models take negligible times (less then a minute per iteration). 					

	
