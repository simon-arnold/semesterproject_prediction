# Load Prediction – Semester Project

This project includes a **complete Conda environment** that installs all required packages for the project (including PyTorch, CUDA, NumPy, Pandas, Matplotlib, Scikit-Learn, etc.).

The environment configuration is stored in the file `environment.yml`.

## Recreate the Environment on Your System

1. Make sure [Conda](https://docs.conda.io/en/latest/) is installed.  
2. Open a terminal and navigate to the project directory.  
3. Create the Conda environment from the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

4. Activate the environment:

   ```bash
   conda activate sp_project
   ```

5. Download and Prepare Electricity Data

   ```bash
   # Create folder and download dataset
   mkdir -p data && cd data
   zenodo_get -r 3581895

   # Install 7-Zip and extract
   sudo apt install p7zip-full
   7z x Data_vs01.7z

   # Clean up
   rm Data_vs01.7z
   rm -rf rawData/
   cd ..
   ```

6. Download and prepare Weather Data
   ```bash
   cd data/data/
   wget -O weather_data_house_a_LUZ.csv https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/luz/ogd-smn_luz_t_historical_2010-2019.csv
   cd ../..
   ```
   
