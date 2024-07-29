# -*- coding: utf-8 -*-
# Argo.py
#------------------------------------------------------------------------------
# Created By: Savannah Stephenson and Hartmut Frenzel
# Creation Date: 07/26/2024
# Version: 0.1 (alpha)
#------------------------------------------------------------------------------
""" The Argo class contains the primary functions for downloading and handling
    data gathered from the Argo Global Data Assebly Centers.
"""
#------------------------------------------------------------------------------
#
#
## Standard Imports
from datetime import datetime, timedelta, timezone
import shutil
import gzip
## Third Party Imports
from pathlib import Path
import requests
import numpy as np
import matplotlib.path as mpltPath
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import netCDF4
# Local Imports
from OneArgoSettings import DownloadSettings, SourceSettings

class Argo:
    """ The Argo class contains the primary functions for downloading and handling
        data gathered from GDAC including a constructor, select_profiels(), 
        trajectories(), load_float_data(), and sections().
    """
    #######################################################################
    # Constructor
    #######################################################################
    def __init__(self, user_settings: str = None) -> None:
        """ The Argo constructor downloads the index files form GDAC and
            stores them in the proper directories defined in the
            DownloadSettings class. It then constructs thee dataframes
            from the argo_synthetic-profile_index.txt file and the
            ar_index_global_prof.txt file for use in class function
            calls. Two of the dataframes are a reflection of the index
            files, the third dataframe is a two column frame with
            float ids and if they are a bgc float or not.
            :param: user_settings : str - An optional parameter that will be used
                to initialize the *Settings classes if passed. This should be the
                full filepath.
            NOTE: If the user has their own settings configuration and has
            set keep_index_in_memory to false then the dataframes will be
            removed from memory at the end of construction and will be
            reloaded with following Argo function calls, meaning that
            functions will take longer but occupy less memory if this
            option is set to false.
        """
        self.download_settings = DownloadSettings(user_settings)
        self.source_settings = SourceSettings(user_settings)
        if self.download_settings.verbose:
            print('Starting initialize process...')
        if self.download_settings.verbose:
            print(f'Your current download settings are: {self.download_settings}')
        if self.download_settings.verbose:
            print(f'Your current source settings are: {self.source_settings}')
        # Check for and create subdirectories if needed
        if self.download_settings.verbose:
            print('Checking for subdirectories...')
        self.__initialize_subdirectories()
        # Download files from GDAC to Index directory
        if self.download_settings.verbose:
            print('\nDownloading index files...')
        for file in self.download_settings.index_files:
            self.__download_file(file)
        # Load the index files into dataframes
        if self.download_settings.verbose:
            print('\nTransferring index files into dataframes...')
        self.sprof_index  = self.__load_sprof_dataframe()
        self.prof_index = self.__load_prof_dataframe()
        # Add column noting if a profile is also in the sprof_index, which is true for bgc floats
        if self.download_settings.verbose:
            print('Marking bgc floats in prof_index dataframe...')
        self.__mark_bgcs_in_prof()
        # Create float_stats reference index for use in select profiles
        if self.download_settings.verbose:
            print('Creating float_stats dataframe...')
        self.float_stats = self.__load_float_stats()
        # Print number of floats
        if self.download_settings.verbose:
            self.__display_floats()
            print('Initialization is finished\n\n')
        if not self.download_settings.keep_index_in_memory:
            if self.download_settings.verbose:
                print('Removing dataframes from memory...')
            del self.sprof_index
            del self.prof_index


    #######################################################################
    # Public Functions
    #######################################################################
    def select_profiles(self, lon_lim: list = [-180, 180], lat_lim: list = [-90, 90],
                        start_date: str = '1995-01-01', end_date: str = None, **kwargs)-> dict:
        """ select_profiles is a public function of the Argo class that returns a
            dictionary if float IDs and profile lists that match the passed criteria.
            :param: lon_lim : list - Longitude limits
            :param: lat_lim : list - Latitude limits
            :param: start_date : str - A UTC date in YYYY-MM-DD format.
            :param: end_date : str - An optional UTC date in YYYY-MM-DD format.
            :param: kargs : keyvalue arguments - Optional key argument values for
                further filtering of the float profiles returned by the function.
            :return: narrowed_profiles : dict - A dictionary with float ID
                keys corresponding to a list of profiles that match criteria.
            NOTE:
            The longitude and latitude limits can be entered as either
            two element lists, in which case the limits will be interpreted
            as maximum and minimum limits tht form a rectangle, or they
            can be entered as a longer list in which case each pair of longitude
            and latitude values correspond to a vertices of a polygon.
            The longitude and latitude limits can be input in any 360 degree
            range that encloses all the desired longitude values.
            Key/argument value options in progress:
            floats=floats[] or float: Select profiles only from these floats that must
                    match all other criteria
            ocean=ocean: Valid choices are 'A' (Atlantic), 'P' (Pacific), and
                    'I' (Indian). This selection is in addition to the specified
                    longitude and latitude limits. (To select all floats and
                    profiles from one ocean basin, leave lon_lim and lat_lim
                    empty.)
            outside='none' or 'time' or 'space' or'both': By default, only float profiles
                    that are within both the temporal and spatial constraints are
                    returned ('none'); specify to also maintain profiles outside
                    the temporal constraints ('time'), spatial constraints
                    ('space'), or both constraints ('both')
            type', type: Valid choices are 'bgc' (select BGC floats only),
                    'phys' (select core and deep floats only),
                    and 'all' (select all floats that match other criteria).
                    If type is not specified, but sensors are, then the type will
                    be set to 'bgc' if sensors other than PRES, PSAL, TEMP, or CNDC
                    are specified.
                    In all other cases the default type is DownloadSettings.float_type,
                    which is set in the Argo constructor, you can also set the float_type
                    as a different value if passing a configuration file to the Argo constructor.
            would like to implement before end of project/easier ones
            sensor='sensor' or [sensors], SENSOR_TYPE: This option allows the selection by
                    sensor type. Available as of 2024: PRES, PSAL, TEMP, DOXY, BBP,
                    BBP470, BBP532, BBP700, TURBIDITY, CP, CP660, CHLA, CDOM,
                    NITRATE, BISULFIDE, PH_IN_SITU_TOTAL, DOWN_IRRADIANCE,
                    DOWN_IRRADIANCE380, DOWN_IRRADIANCE412, DOWN_IRRADIANCE443,
                    DOWN_IRRADIANCE490, DOWN_IRRADIANCE555, DOWN_IRRADIANCE670,
                    UP_RADIANCE, UP_RADIANCE412, UP_RADIANCE443, UP_RADIANCE490,
                    UP_RADIANCE555, DOWNWELLING_PAR, CNDC, DOXY2, DOXY3, BBP700_2
                    Multiple sensors can be entered as a list, e.g.: ['DOXY';'NITRATE']
            dac=dac: Select by Data Assimilation Center responsible for the floats.
                    A single DAC can be entered as a string (e.g.: 'aoml'),
                    multiple DACs can be entered as a list of strings (e.g.:
                    ['meds';'incois'].
                    Valid values as of 2024 are any: {'aoml'; 'bodc'; 'coriolis'; ...
                    'csio'; 'csiro'; 'incois'; 'jma'; 'kma'; 'kordi'; 'meds'}
        """
        if self.download_settings.verbose:
            print('Starting select_profiles...')
        self.epsilon = 1e-3
        self.lon_lim = lon_lim
        self.lat_lim = lat_lim
        self.start_date = start_date
        self.end_date = end_date
        self.outside = kwargs.get('outside')
        self.float_type = kwargs.get('type') if kwargs.get('type') is not None \
            else self.download_settings.float_type
        self.float_ids = kwargs.get('floats')
        self.ocean = kwargs.get('ocean')
        self.sensor = kwargs.get('sensor')
        if self.download_settings.verbose:
            print('Validating parameters...')
        self.__validate_lon_lat_limits()
        self.__validate_start_end_dates()
        if self.outside:
            self.__validate_outside_kwarg()
        if self.float_type:
            self.__validate_type_kwarg()
        if self.ocean:
            self.__validate_ocean_kwarg()
        # if self.sensor : self.__validate_sensor_kwarg()
        # Load correct dataframes according to self.float_type and self.float_ids
        # we set self.selected_from_sprof_index and self.selected_from_prof_index
        # in this function which will be used in __narrow_profiles_by_criteria
        self.__prepare_selection()
        # Narrow down float profiles and save in dictionary
        narrowed_profiles = self.__narrow_profiles_by_criteria()
        if not self.download_settings.keep_index_in_memory:
            if self.download_settings.verbose:
                print('Removing dataframes from memory...')
            del self.sprof_index
            del self.prof_index
            del self.selection_frame
        if self.download_settings.verbose:
            print(f'Floats Selected: {narrowed_profiles.keys()}\n')
        return narrowed_profiles


    def trajectories(self, floats: int | list | dict, visible: bool = True,
                     save_to: str = None)-> None:
        """ This function plots the trajectories of one or more specified float(s)
            :param: floats : int | list | dict - Floats to plot.
            :param: visible : bool - A boolean value determining if the trajectories
                plot is shown to the user through a popup window.
            :param: save_to : str - A path to a folder where the user would like
                to save the trajectories plot(s). The path must exist.
                The file name is automatically generated.
        """
        # Validate save_to file path
        if save_to is not None:
            save_to = Path(save_to)
            self.__validate_plot_save_path(Path(save_to))
        # Check that dataframes are loaded into memory
        if not self.download_settings.keep_index_in_memory:
            self.sprof_index = self.__load_sprof_dataframe()
            self.prof_index = self.__load_prof_dataframe()
        # Validate passed floats
        self.float_ids = floats
        self.__validate_floats_kwarg()
        # Pull rows/profiles for passed floats
        floats_profiles = self.__filter_by_floats()
        # If keep index in memory is false remove other dataframes
        if not self.download_settings.keep_index_in_memory:
            if self.download_settings.verbose:
                print('Removing dataframes from memory...')
            del self.sprof_index
            del self.prof_index
        # Set up basic graph size
        fig = plt.figure(figsize=(10, 10))
        # Define the median longitude for the graph to be centered on
        lons = floats_profiles['longitude'].dropna().values.tolist()
        sorted_lons = np.sort(lons)
        median_lon = np.nanmedian(sorted_lons)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=median_lon))
        # Add landmasses and coastlines
        ax.add_feature(cf.COASTLINE, linewidth=1.5)
        ax.add_feature(cf.LAND, zorder=2, edgecolor='k', facecolor='lightgray')
        # Plot trajectories of passed floats with colorblind friendly pallet
        colors = ("#56B4E9", "#009E73", "#F0E442", "#0072B2",
                  "#CC79A7", "#D55E00", "#E69F00", "#000000")
        for i, float_id in enumerate(self.float_ids):
            specific_float_profiles = floats_profiles[floats_profiles['wmoid'] == float_id]
            ax.plot(specific_float_profiles['longitude'].values,
                    specific_float_profiles['latitude'].values,
                    marker='.', alpha=0.7, linestyle='-', linewidth=2, transform=ccrs.Geodetic(),
                    label=f'Float {float_id}', color=colors[i % len(colors)])
        # Set graph limits based on passed points
        self.__set_graph_limits(ax, 'x')
        self.__set_graph_limits(ax, 'y')
        # Add grid lines
        self.__add_grid_lines(ax)
        # Add Legend outside of the main plot
        if len(self.float_ids) > 1:
            plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        # Setting Title
        if len(self.float_ids) == 1:
            ax.set_title(f'Trajectory of {self.float_ids[0]}', fontsize=18, fontweight='bold')
        elif len(self.float_ids) < 4:
            ax.set_title(f'Trajectories of {self.float_ids}', fontsize=18, fontweight='bold')
        else:
            ax.set_title('Trajectories of Selected Floats', fontsize=18, fontweight='bold')
        plt.tight_layout();

        # Saving Graph
        if save_to is not None:
            if len(self.float_ids) == 1:
                save_path = save_to.joinpath(f'trajectories_{self.float_ids}[0]')
            else:
                save_path = save_to.joinpath(f'trajectories_plot_{len(self.float_ids)}_floats')
            plt.savefig(f'{save_path}')

        # Displaying graph
        if visible:
            plt.show()


    def load_float_data(self, floats: int | list | dict, variables: str | list = None)-> pd:
        """ A function to load float data into memory.
            :param: floats : int | list | dict - A float or list of floats to
                load data from. Or a dictionary specifying floats and profiles
                to read from the .nc file.
            :param: variables : str | list - An optional parameter to list variables
                that the user would like included in the dataframe. If the variable is not
                in the float passed then only the surface level of the profile will be included.
            :return: float_data : pd - A dataframe with requested float data.
        """
        # Check that index files are in memory
        if not self.download_settings.keep_index_in_memory:
            self.sprof_index = self.__load_sprof_dataframe()
            self.prof_index = self.__load_prof_dataframe()
        # Check that passed float is inside of the dataframes
        self.float_ids = floats
        self.__validate_floats_kwarg()
        # Validate passed variables
        self.float_variables = variables
        if self.float_variables:
            self.__validate_float_variables_arg()
        # Check if the user has passed only phys float variables
        if self.float_variables is not None:
            phys_variables = ['TEMP', 'PSAL', 'PRES', 'CNDC']
            only_phys = all(x in phys_variables for x in self.float_variables)
        else:
            only_phys = False
        # Download .nc files for passed floats
        files = []
        for wmoid in self.float_ids:
            # If the float is a phys float, or if the user has provided no variables
            # or only phys variables then then use the corresponding prof file
            if ((not self.float_stats.loc[self.float_stats['wmoid'] == wmoid, 'is_bgc'].values[0])
                or (self.float_variables is None) or (only_phys)):
                file_name = f'{wmoid}_prof.nc'
                files.append(file_name)
            # If the float is a bgc float it will have a corresponding sprof file
            else:
                file_name = f'{wmoid}_Sprof.nc'
                files.append(file_name)
            # Download file
            self.__download_file(file_name)
        # Read from nc files into dataframe
        float_data_frame = self.__fill_float_data_dataframe(files)
        return float_data_frame


    def sections(self, float_data: pd, variables: str | list, visible: bool = True,
                 save_to: str = None)-> None:
        """ A function to graph section plots for the passed variables using data
            from the passed float_data dataframe.
            :param: float_data : pd - A dataframe created from load_float_data
                that contains data pulled from .nc files.
            :param: variables : str or list - The variable(s) the user would
                like section plots of.
            :param: visible : bool - A boolean value determining if the section
                plot is shown to the user through a popup window.
            :param: save_to : str - A path to a folder where the 
                user would like to save the section plot(s). The folder must exist.
                The filename is automatically generated.
        """
        # Validate passed variables
        self.float_variables = variables
        self.__validate_float_variables_and_permutations_arg()
        # Validate passed dataframe
        self.float_data = float_data
        self.__validate_float_data_dataframe()
        # Validate save_to file path
        if save_to is not None:
            save_to = Path(save_to)
            self.__validate_plot_save_path(save_to)
        # Determine Unique WMOID
        unique_float_ids = self.float_data['WMOID'].unique()
        # Make one plot for each float/variable combination
        for float_id in unique_float_ids:
            filtered_df = self.float_data[self.float_data['WMOID'] == float_id]
            # Getting unique profile values for the current float
            unique_values = filtered_df['CYCLE_NUMBER'].unique()
            # Check that the float has more than one profile (more than one cycle number)
            if len(unique_values) == len(filtered_df):
                if self.download_settings.verbose:
                    print(f'Float {float_id} has only one profile, skipping this float...')
                continue
            if self.download_settings.verbose:
                print(f'Generating section plots for float {float_id}...')
            for variable in self.float_variables:
                # Pulling column for current float and variable
                float_variable_data = filtered_df[variable]
                # Check that the float actually has data for the passed variable
                if float_variable_data.isna().all():
                    if self.download_settings.verbose:
                        print(f'Float {float_id} has no data for variable {variable}, ' +
                              'skipping plot...')
                    continue
                # Otherwise plot the section
                if self.download_settings.verbose:
                    print(f'Generating {variable} section plot for float {float_id}...')
                self.__plot_section(self.float_data, float_id, variable, visible, save_to)


    #######################################################################
    # Private Functions
    #######################################################################
    def __initialize_subdirectories(self) -> None:
        """ A function that checks for and creates the necessary folders as
            listed in the download settings sub_dir list.
        """
        for directory in self.download_settings.sub_dirs:
            directory_path = self.download_settings.base_dir.joinpath(directory)
            if directory_path.exists():
                if self.download_settings.verbose:
                    print(f'The {directory_path} directory already exists')
            else:
                try:
                    if self.download_settings.verbose:
                        print(f'Creating the {directory} directory')
                    directory_path.mkdir()
                except OSError as e:
                    if self.download_settings.verbose:
                        print(f'Failed to create the {directory} directory: {e}')


    def __download_file(self, file_name: str) -> None:
        """ A function to download and save an index file from GDAC sources.
            :param: filename : str - The name of the file we are downloading.
        """
        if file_name.endswith('.txt'):
            directory = Path(self.download_settings.base_dir.joinpath("Index"))
        elif file_name.endswith('.nc'):
            directory = Path(self.download_settings.base_dir.joinpath("Profiles"))
        # Get the expected filepath for the file
        file_path = directory.joinpath(file_name)
        # Check if the filepath exists
        if file_path.exists():
            # Check if .txt file needs to be updated
            if file_name.endswith('.txt') :
                # Check if the settings allow for updates of index files
                if self.download_settings.update == 0:
                    if self.download_settings.verbose:
                        print('The download settings have update set to 0, ' +
                              'indicating index files will not be updated.')
                else:
                    last_modified_time = Path(file_path).stat().st_mtime
                    current_time = datetime.now().timestamp()
                    seconds_since_modified = current_time - last_modified_time
                    # Check if the file should be updated
                    if seconds_since_modified > self.download_settings.update:
                        if self.download_settings.verbose:
                            print(f'Updating {file_name}...')
                        self.__try_download(file_name ,True)
                    else:
                        if self.download_settings.verbose:
                            print(f'{file_name} does not need to be updated yet.')
           # Check if .nc file needs to be updated
            elif file_name.endswith('.nc'):
                # Check if the file should be updated using function
                if self.__check_nc_update(file_path, file_name):
                    if self.download_settings.verbose:
                        print(f'Updating {file_name}...')
                    self.__try_download(file_name ,True)
                else:
                    if self.download_settings.verbose:
                        print(f'{file_name} does not need to be updated yet.')
        # if the file doesn't exist then download it
        else:
            if self.download_settings.verbose:
                print(f'{file_name} needs to be downloaded.')
            self.__try_download(file_name, False)


    def __check_nc_update(self, file_path: Path, file_name: str)-> bool:
        """ A function to check if an .nc file needs to be updated.
            :param: file_path : Path - The file_path for the .nc file we
                are checking for update.
            :param: file_name : str - The name of the .nc file.
            :return: update_status : bool - A boolean value indicating
                that the passed file should be updated.
        """
        # Pull float id from file_name
        float_id = file_name.split('_')[0]
        # Get float's latest update date
        if (self.prof_index.loc[self.prof_index['wmoid'] == int(float_id), 'is_bgc'].any()
            and file_name.endswith('_prof.nc')):
            # Use the prof update date for the bgc float because user didn't pass any bgc sensors
            dates_for_float = self.prof_index[self.prof_index['wmoid'] == int(float_id)]
            index_update_date = pd.to_datetime( \
                dates_for_float['date_update'].drop_duplicates().max())
        else:
            index_update_date = pd.to_datetime( \
                self.float_stats.loc[self.float_stats['wmoid'] == int(float_id),
                                     'date_update'].iloc[0])
        # Read DATE_UPDATE from .nc file
        nc_file = netCDF4.Dataset(file_path, mode='r')
        netcdf_update_date = nc_file.variables['DATE_UPDATE'][:]
        nc_file.close()
        # Convert the byte strings of file_update_date into a regular string
        julian_date_str = b''.join(netcdf_update_date).decode('utf-8')
        netcdf_update_date = datetime.strptime(julian_date_str,
                                               '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        netcdf_update_date = np.datetime64(netcdf_update_date)
        # If the .nc file's update date is less than
        # the date in the index file return true
        # indicating that the .nc file must be updated
        # otherwise return false
        return bool(netcdf_update_date < index_update_date)


    def __try_download(self, file_name: str, update_status: bool)-> None:
        """ A function that attempts to download a file from both GDAC sources.
            :param: file_name : str - The name of the file to download
            :param: update_status: bool - True if the file exists and we
                are trying to update it. False if the file hasn't been
                downloaded yet.
        """
        if file_name.endswith('.txt'):
            directory = Path(self.download_settings.base_dir.joinpath("Index"))
            first_save_path = directory.joinpath("".join([file_name, ".gz"]))
            second_save_path = directory.joinpath(file_name)
        elif file_name.endswith('.nc'):
            directory = Path(self.download_settings.base_dir.joinpath("Profiles"))
            first_save_path = directory.joinpath(file_name)
            second_save_path = None
        success = False
        iterations = 0
        # Determining float id if file is an .nc file
        if file_name.endswith('.nc'):
            # Extract float id from filename
            float_id = file_name.split('_')[0]
            # Extract dac for that float id from datafrmae
            filtered_df = self.prof_index[self.prof_index['wmoid'] == int(float_id)]
            dac = filtered_df['dacs'].iloc[0]
            # Add trailing forward slashes for formating
            dac = f'{dac}/'
            float_id = f'{float_id}/'
        while (not success) and (iterations < self.download_settings.max_attempts):
            # Try both hosts (preferred one is listed first in SourceSettings)
            for host in self.source_settings.hosts:
                if file_name.endswith('.txt'):
                    url = "".join([host, file_name, ".gz"])
                elif file_name.endswith('.nc'):
                    url = "".join([host,'dac/', dac, float_id, file_name])
                if self.download_settings.verbose:
                    print(f'Downloading {file_name} from {url}...')
                try:
                    with requests.get(url, stream=True,
                                      timeout=self.download_settings.timeout) as r:
                        r.raise_for_status()
                        with open(first_save_path, 'wb') as f:
                            r.raw.decode_content = True
                            shutil.copyfileobj(r.raw, f)
                    if second_save_path is not None:
                        # If the file has a second save path it was first downloaded as a .gz file
                        # so it must be unzipped.
                        if self.download_settings.verbose:
                            print(f'Unzipping {file_name}.gz...')
                        with gzip.open(first_save_path, 'rb') as gz_file:
                            with open(second_save_path, 'wb') as txt_file:
                                shutil.copyfileobj(gz_file, txt_file)
                        # Remove extraneous .gz file
                        first_save_path.unlink()
                        success = True
                    elif file_name.endswith('.nc'):
                        # Check that the file can be read, only keep download if file can be read
                        try:
                            nc_file = netCDF4.Dataset(first_save_path, mode='r')
                            nc_file.close()
                            success = True
                        except OSError:
                            # The file could not be read
                            if self.download_settings.verbose:
                                print(f'{first_save_path} cannot be read; trying again...')
                    if success:
                        if self.download_settings.verbose:
                            print('Success!')
                        # Exit the loop if download is successful so we don't try additional
                        # sources for no reason
                        break
                except requests.RequestException as e:
                    print(f'Error encountered: {e}. Trying next host...')
            # Increment Iterations
            iterations += 1
        # If ultimately nothing could be downloaded
        if not success:
            if update_status:
                print(f'WARNING: Update of {file_name} failed, you are working with outdated data.')
            else:
                raise OSError('Download failed!' +
                                f'{file_name} could not be downloaded at this time.')


    def __load_sprof_dataframe(self) -> pd:
        """ A function to load the sprof index file into a dataframe for easier reference.
        """
        file_name = "argo_synthetic-profile_index.txt"
        file_path = Path.joinpath(self.download_settings.base_dir, 'Index', file_name)
        # There are 8 header lines in both index files
        sprof_index = pd.read_csv(file_path, delimiter=',', header=8,
                                  parse_dates=['date','date_update'], date_format='%Y%m%d%H%M%S')
        # Parsing out variables in first column: file
        dacs = sprof_index['file'].str.split('/').str[0]
        sprof_index.insert(1, "dacs", dacs)
        wmoid = sprof_index['file'].str.split('/').str[1].astype('int')
        sprof_index.insert(0, "wmoid", wmoid)
        profile = sprof_index['file'].str.split('_').str[1].str.replace('.nc', '')
        sprof_index.insert(2, "profile", profile)
        # Splitting the parameters into their own columns
        parameters_split = sprof_index['parameters'].str.split()
        data_types_split = sprof_index['parameter_data_mode'].apply(list)
        # R: raw data, A: adjusted mode (real-time adjusted),
        # D: delayed mode quality controlled
        data_type_mapping = {np.nan: 0, 'R':1, 'A':2, 'D':3 }
        mapped_data_types_split = data_types_split.apply(lambda lst: [data_type_mapping.get(x, 0)
                                                                      if pd.notna(x) else 0
                                                                      for x in lst])
        # Create a new DataFrame from the split parameters
        expanded_df = pd.DataFrame({
            'index': sprof_index.index.repeat(parameters_split.str.len()),
            'parameter': parameters_split.explode(),
            'data_type': mapped_data_types_split.explode()
        })
        # Pivot the expanded DataFrame to get parameters as columns
            # Line here to suppress warning about fillna()
            # being depreciated in future versions of pandas:
            # with pd.option_context('future.no_silent_downcasting', True):
        result_df = expanded_df.pivot(index='index', columns='parameter', \
            values='data_type').fillna(0).infer_objects(copy=False).astype('int8')
        # Fill in source_settings information based off of sprof index file before removing rows
        if self.download_settings.verbose:
            print('Filling in source settings information...')
        self.source_settings.set_avail_vars(sprof_index)
        # Merge the pivoted DataFrame back with the original DataFrame and drop split rows
        if self.download_settings.verbose:
            print('Marking Parameters with their data mode...')
        sprof_index = sprof_index.drop(columns=['parameters', 'parameter_data_mode'])
        sprof_index = sprof_index.join(result_df)
        # Add profile_index column
        sprof_index.sort_values(by=['wmoid', 'date'], inplace=True)
        sprof_index.insert(0, "profile_index", 0)
        sprof_index['profile_index'] = sprof_index.groupby('wmoid')['date'].cumcount() + 1
        return sprof_index


    def __load_prof_dataframe(self) -> pd:
        """ A function to load the prof index file into a dataframe for easier reference.
        """
        file_name = "ar_index_global_prof.txt"
        file_path = Path.joinpath(self.download_settings.base_dir, 'Index', file_name)
        # There are 8 header lines in this index file
        prof_index = pd.read_csv(file_path, delimiter=',', header=8,
                                 parse_dates=['date','date_update'], date_format='%Y%m%d%H%M%S')
        # Splitting up parts of the first column
        dacs = prof_index['file'].str.split('/').str[0]
        prof_index.insert(0, "dacs", dacs)
        wmoid = prof_index['file'].str.split('/').str[1].astype('int')
        prof_index.insert(1, "wmoid", wmoid)
        d_file = prof_index['file'].str.split('/').str[3].str.startswith('D')
        prof_index.insert(2, "D_file", d_file)
        # Add profile_index column
        prof_index.sort_values(by=['wmoid', 'date'], inplace=True)
        prof_index.insert(0, "profile_index", 0)
        prof_index['profile_index'] = prof_index.groupby('wmoid')['date'].cumcount() + 1
        # Fill in source_settings information based off of sprof index file before removing rows
        if self.download_settings.verbose:
            print('Filling in source settings information...')
        self.source_settings.set_dacs(prof_index)
        return prof_index


    def __mark_bgcs_in_prof(self):
        """ A function to mark whether the floats listed in prof_index are
            biogeochemical floats or not.
        """
        bgc_floats = self.sprof_index['wmoid'].unique()
        is_bgc = self.prof_index['wmoid'].isin(bgc_floats)
        self.prof_index.insert(1, "is_bgc", is_bgc)


    def __load_float_stats(self)-> pd:
        """ Function to create a dataframe with float IDs,
            their is_bgc status, and their most recent update
            date for use in select_profiles().
            Data for physical floats are taken from the prof index
            file and data for BGC floats are taken from the Sprof index file.
        """
        # Dataframe with wmoid and date updated for both prof and sprof
        float_bgc_status_prof = self.prof_index.loc[~self.prof_index['is_bgc'], ['wmoid',
                                                                                 'date_update']]
        float_bgc_status_sprof = self.sprof_index[['wmoid', 'date_update']]
        # Only keeping rows with most recent date updated
        floats_stats_prof = float_bgc_status_prof.groupby('wmoid',
                                                          as_index=False)['date_update'].max()
        floats_stats_sprof = float_bgc_status_sprof.groupby('wmoid',
                                                            as_index=False)['date_update'].max()
        # Adding the is_bgc column
        floats_stats_sprof['is_bgc'] = True
        floats_stats_prof['is_bgc'] = False
        # Combining the two dataframes for one refrence frame for all floats
        floats_stats = pd.concat([floats_stats_sprof, floats_stats_prof]).sort_values(by='wmoid')
        return floats_stats


    def __display_floats(self) -> None:
        """ A function to display information about the number of floats initially
            observed in the unfiltered dataframes.
        """
        floats = self.prof_index['wmoid'].unique()
        profiles = self.prof_index['file'].unique()
        print(f"\n{len(floats)} floats with {len(profiles)} profiles found.")
        bgc_floats = self.sprof_index['wmoid'].unique()
        profiles = self.sprof_index['file'].unique()
        print(f"{len(bgc_floats)} BGC floats with {len(profiles)} profiles found.")


    def __validate_lon_lat_limits(self)-> None:
        """ Function to validate the length, order, and contents of
            longitude and latitude limits passed to select_profiles.
        """
        if self.download_settings.verbose:
            print('Validating longitude and latitude limits...')
        # Validating Lists
        if len(self.lon_lim) != len(self.lat_lim):
            raise KeyError('The length of the longitude and latitude lists must be equal.')
        if len(self.lon_lim) == 2:
            if (self.lon_lim[1] <= self.lon_lim[0]) or (self.lat_lim[1] <= self.lat_lim[0]):
                if self.download_settings.verbose:
                    print(f'Longitude Limits: min={self.lon_lim[0]} max={self.lon_lim[1]}')
                    print(f'Latitude Limits: min={self.lat_lim[0]} max={self.lat_lim[1]}')
                raise KeyError('When passing longitude and latitude lists using the [min, max] ' +
                               'format, the max value must be greater than the min value.')
            if ((abs(self.lon_lim[1] - self.lon_lim[0] - 360.0) < self.epsilon) and
                (abs(self.lat_lim[1] - self.lat_lim[0] - 180.0) < self.epsilon)):
                self.keep_full_geographic = True
            else:
                self.keep_full_geographic = False
        # Validating latitudes
        if not all(-90 <= lat <= 90 for lat in self.lat_lim):
            print(f'Latitudes: {self.lat_lim}')
            raise KeyError('Latitude values should be between -90 and 90.')
        # Validate Longitudes
        # Checking range of longitude values
        lon_range = max(self.lon_lim) - min(self.lon_lim)
        if lon_range > 360 or lon_range <= 0:
            if self.download_settings.verbose:
                print(f'Current longitude range: {lon_range}')
            raise KeyError('The range between the maximum and minimum longitude values must be ' +
                           'between 0 and 360.')
        # Adjusting values to fit between -180 and 360
        if  min(self.lon_lim) < -180:
            if self.download_settings.verbose:
                print('Adjusting within -180')
            self.lon_lim = [lon + 360.00 for lon in self.lon_lim]


    def __validate_start_end_dates(self):
        """ A function to validate the start and end date strings passed to select_profiles and
            converts them to datetimes for easier comparison to dataframe values later on.
        """
        if self.download_settings.verbose:
            print('Validating start and end dates...')
        # Parse Strings to Datetime Objects
        try:
            # Check if the string matches the expected format
            self.start_date = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
            # end_date is optional and should be set to tomorrow if not provided
            if self.end_date is not None:
                self.end_date = datetime.fromisoformat(self.end_date).replace(tzinfo=timezone.utc)
            else:
                self.end_date = datetime.now(timezone.utc) + timedelta(days=1)
        except ValueError:
            print(f" Start date: {self.start_date} or end date: {self.end_date} is not in the " +
                  "expected format 'yyyy-mm-dd'")
        # Validate datetimes
        if self.start_date > self.end_date:
            if self.download_settings.verbose:
                print(f'Current start date: {self.start_date}')
                print(f'Current end date: {self.end_date}')
            raise ValueError('The start date must be before the end date.')
        if self.start_date < datetime(1995, 1, 1, tzinfo=timezone.utc):
            if self.download_settings.verbose:
                print(f'Current start date: {self.start_date}')
            raise ValueError('Start date must be after at least: ' +
                             f'{datetime(1995, 1, 1, tzinfo=timezone.utc)}.')
        # Set to datetime64 for dataframe comparisons
        self.start_date = np.datetime64(self.start_date)
        self.end_date = np.datetime64(self.end_date)


    def __validate_outside_kwarg(self):
        """ A function to validate the value of the
            optional 'outside' keyword argument.
        """
        if self.download_settings.verbose:
            print("Validating 'outside' keyword argument...")
        if self.outside is not None:
            if self.outside not in ('time', 'space', 'both'):
                raise KeyError("The only acceptable values for the 'outside' keyword argument " +
                               "are 'time', 'space', and 'both'.")


    def __validate_type_kwarg(self):
        """ A function to validate the value of the
            optional 'type' keyword argument.
        """
        if self.download_settings.verbose:
            print("Validating 'type' keyword argument...")
        if self.float_type not in ('all', 'phys', 'bgc'):
            raise KeyError("The only acceptable values for the 'type' keyword argument are 'all'," +
                           " 'phys', and 'bgc'.")


    def __validate_floats_kwarg(self):
        """ A function to validate the 'floats' keyword argument.
            The 'floats' must be a list even if it is a single float.
            If the floats passed are in a dictionary we separate the keys
            from the dictionary for flexibility.
        """
        if self.download_settings.verbose:
            print("Validating passed floats...")
        # If user has passed a dictionary
        if isinstance(self.float_ids, dict):
            self.float_profiles_dict = self.float_ids
            self.float_ids = list(self.float_ids.keys())
        # If user has passed a single float
        elif not isinstance(self.float_ids, list):
            self.float_profiles_dict = None
            self.float_ids = [self.float_ids]
        # If user has passed a list
        else:
            self.float_profiles_dict = None
        # Finding float IDs that are not present in the index dataframes
        missing_floats = [float_id for float_id in self.float_ids if float_id not in
                          self.prof_index['wmoid'].values]
        if missing_floats:
            raise KeyError("The following float IDs do not exist in the dataframes: " +
                           f"{missing_floats}")


    def __validate_ocean_kwarg(self):
        """ A function to validate the value of the
            optional 'ocean' keyword argument.
        """
        if self.download_settings.verbose:
            print("Validating 'ocean' keyword argument...")
        if self.ocean not in ('A', 'P', 'I'):
            raise KeyError("The only acceptable values for the 'ocean' keyword argument are 'A' " +
                           "(Atlantic), 'P' (Pacific), and 'I' (Indian).")


    def __validate_float_variables_arg(self):
        """ A function to validate the value of the
            optional 'variables' passed to
            load_float_data.
        """
        if self.download_settings.verbose:
            print("Validating passed 'variables'...")
        # If user has passed a single variable convert to list
        if not isinstance(self.float_variables, list):
            self.float_variables = [self.float_variables]
        # Finding variables that are not present avaliable variables list
        nonexistent_vars = [x for x in self.float_variables if x not in
                            self.source_settings.avail_vars]
        if nonexistent_vars:
            raise KeyError("The following variables do not exist in the dataframes: " +
                           f"{nonexistent_vars}")


    def __validate_float_variables_and_permutations_arg(self):
        """ A function to validate the value of the
            optional 'variables' passed to
            load_float_data.
        """
        if self.download_settings.verbose:
            print("Validating passed 'variables'...")
        # If user has passed a single variable convert to list
        if not isinstance(self.float_variables, list):
            self.float_variables = [self.float_variables]
        # Constructing list of variables avaliable for plotting
        adjusted_variables = []
        for variable in self.source_settings.avail_vars:
            adjusted_variables.append(variable + '_ADJUSTED')
            adjusted_variables.append(variable + '_ADJUSTED_ERROR')
        available_variables = self.source_settings.avail_vars + adjusted_variables
        # Finding variables that are not present in the available variables list
        nonexistent_vars = [x for x in self.float_variables if x not in available_variables]
        if nonexistent_vars:
            raise KeyError("The following variables do not exist in the dataframes: " +
                           f"{nonexistent_vars}")


    def __validate_float_data_dataframe(self):
        """ A function to validate a dataframe passed
            to sections() so ensure that it has the
            expected columns for graphing section
            plots.
        """
        if self.download_settings.verbose:
            print("Validating passed float_data_dataframe...")
        # Check that the dataframe at the very least has wmoid and variable columns
        required_columns = ['WMOID'] + self.float_variables
        # Identify missing columns
        missing_columns = set(required_columns) - set(self.float_data.columns)
        if missing_columns:
            raise KeyError("The following columns are missing from the dataframe: " +
                           f"{missing_columns}")


    def __validate_plot_save_path(self, save_path: Path):
        """ A function to validate that the save path passed
            actually exists. 
        """
        if not save_path.exists():
            print(f'{save_path} not found')
            raise FileNotFoundError


    def __prepare_selection(self):
        """ A function that determines what dataframes will be loaded/used
            when selecting floats. We determine what dataframes to load
            based on two factors: type and passed floats.
            If type is 'phys', the dataframe based on
            ar_index_global_prof.txt will be used.
            If type is 'bgc', the dataframe based on
            argo_synthetic-profile_index.txt will be used.
            If type is 'all', both dataframes are used.
            BGC floats are taken from argo_synthetic-profile_index.txt,
            non-BGC floats from ar_index_global_prof.txt.
            If the user passed floats, we only load the passed floats
            into the selection frames.
            If keep_index_in_memory is set to false, the dataframes created
            during Argo's constructor are deleted. In this function we only
            reload the necessary dataframes into memory.
        """
        if self.download_settings.verbose:
            print('Preparing float data for filtering...')
        selected_floats_phys = None
        selected_floats_bgc = None
        # Load dataframes into memory if they are not there
        if not self.download_settings.keep_index_in_memory:
            self.sprof_index = self.__load_sprof_dataframe()
            self.prof_index = self.__load_prof_dataframe()
        # We can only validate floats after the dataframes are loaded into memory
        if self.float_ids:
            self.__validate_floats_kwarg()
        # If we aren't filtering from specific floats assign selected frames
        # to the whole index frames
        if self.float_ids is None:
            self.selected_from_prof_index = self.prof_index[~self.prof_index['is_bgc']]
            self.selected_from_sprof_index = self.sprof_index
        # If we do have specific floats to filter from, assign
        # selected floats by pulling those floats from the
        # larger dataframes, only adding floats that match the
        # type to the frames.
        else:
            # Empty default dataframes are needed for the len function below
            self.selected_from_prof_index = pd.DataFrame({'wmoid': []})
            self.selected_from_sprof_index = pd.DataFrame({'wmoid': []})
            if self.float_type != 'phys':
                # Make a list of bgc floats that the user wants
                bgc_filter = ((self.float_stats['wmoid'].isin(self.float_ids)) &
                              (self.float_stats['is_bgc'] == True))
                selected_floats_bgc = self.float_stats[bgc_filter]['wmoid'].tolist()
                # Gather bgc profiles for these floats from sprof index frame
                self.selected_from_sprof_index = \
                    self.sprof_index[self.sprof_index['wmoid'].isin(selected_floats_bgc)]
            if self.float_type != 'bgc':
                # Make a list of phys floats that the user wants
                phys_filter = ((self.float_stats['wmoid'].isin(self.float_ids)) &
                               (self.float_stats['is_bgc'] == False))
                selected_floats_phys = self.float_stats[phys_filter]['wmoid'].tolist()
                # Gather phys profiles for these floats from prof index frame
                self.selected_from_prof_index = \
                    self.prof_index[self.prof_index['wmoid'].isin(selected_floats_phys)]
        if self.download_settings.verbose:
            num_unique_floats = len(self.selected_from_sprof_index['wmoid'].unique()) + \
                len(self.selected_from_prof_index['wmoid'].unique())
            print(f"Filtering through {num_unique_floats} floats")
            num_profiles = len(self.selected_from_sprof_index) + len(self.selected_from_prof_index)
            print(f'There are {num_profiles} profiles associated with these floats\n')


    def __narrow_profiles_by_criteria(self)-> dict:
        """ A function to narrow down the available profiles to only those
            that meet the criteria passed to select_profiles.
            :return: narrowed_profiles : dict - A dictionary with float ID
                keys corresponding to a list of profiles that match criteria.
        """
        # Filter by time, space, and type constraints first.
        if self.float_type == 'bgc' or self.selected_from_prof_index.empty:
            # Empty df for concat
            self.selection_frame_phys = pd.DataFrame()
        else:
            self.selection_frame_phys = \
                self.__get_in_time_and_space_constraints(self.selected_from_prof_index)
        if self.float_type == 'phys' or self.selected_from_sprof_index.empty:
            # Empty df for concat
            self.selection_frame_bgc = pd.DataFrame()
        else:
            self.selection_frame_bgc = \
                self.__get_in_time_and_space_constraints(self.selected_from_sprof_index)
        # Set the selection frame
        self.selection_frame = pd.concat([self.selection_frame_bgc, self.selection_frame_phys])
        # Remove extraneous frames
        if not self.download_settings.keep_index_in_memory:
            del self.sprof_index
            del self.prof_index
        del self.selection_frame_bgc
        del self.selection_frame_phys
        if self.selection_frame.empty:
            if self.download_settings.verbose:
                print('No matching floats found')
            return {}
        if self.download_settings.verbose:
            print(f"{len(self.selection_frame['wmoid'].unique())} floats selected")
            print(f'{len(self.selection_frame)} profiles selected according to time and space ' +
                  'constraints')
        # Filter by other constraints, these functions will use self.selection_frame
        # so we don't have to pass a frame
        if self.ocean:
            self.__get_in_ocean_basin()
        # other narrowing functions that act on created selection frame...
        # Convert the working dataframe into a dictionary
        selected_floats_dict = self.__dataframe_to_dictionary()
        return selected_floats_dict


    def __get_in_geographic_range(self, dataframe_to_filter: pd)-> list:
        """ A function to create and return a true false array indicating
            profiles that fall within the geographic range.
        """
        # If the user has passed us the entire globe don't go through the whole
        # process of checking if the points of all the floats are inside the polygon
        if self.keep_full_geographic:
            return  [True] * len(dataframe_to_filter)
        if self.download_settings.verbose:
            print('Sorting floats for those within the geographic range...')
        # Make points out of profile lat and lons
        if self.download_settings.verbose:
            print('Creating point list from profiles...')
        profile_points = np.empty((len(dataframe_to_filter), 2))
        # The longitudes in the dataframe are standardized to fall within -180 and 180.
        # but our longitudes only have a standard minimum value of -180. In this section
        # we adjust the longitude and latitudes in the dataframe to follow this minimum
        # only approach.
        if max(self.lon_lim) > 180:
            if self.download_settings.verbose:
                print(f'The max value in lon_lim is {max(self.lon_lim)}')
                print('Adjusting longitude values...')
            profile_points[:,0] = dataframe_to_filter['longitude'].apply(lambda x: x + 360
                                                                         if -180 < x <
                                                                         min(self.lon_lim)
                                                                         else x).values
        else:
            profile_points[:,0] = dataframe_to_filter['longitude'].values
        # Latitudes in the dataframe are good to go
        profile_points[:,1] = dataframe_to_filter['latitude'].values
        # Create polygon or box using lat_lim and lon_lim
        if self.download_settings.verbose:
            print('Creating polygon...')
        if len(self.lat_lim) == 2:
            shape = [[max(self.lon_lim), min(self.lat_lim)], # Top-right
                     [max(self.lon_lim), max(self.lat_lim)], # Bottom-right
                     [min(self.lon_lim), max(self.lat_lim)], # Bottom-left
                     [min(self.lon_lim), min(self.lat_lim)]] # Top-left
        else:
            shape = []
            for lon, lat in zip(self.lon_lim, self.lat_lim):
                shape.append([lon, lat])
        # Define a t/f array for profiles within the shape
        path = mpltPath.Path(shape)
        profiles_in_range = path.contains_points(profile_points)
        if self.download_settings.verbose:
            profiles_in_range_dataframe = dataframe_to_filter[profiles_in_range]
            print(f"{len(profiles_in_range_dataframe['wmoid'].unique())} floats fall within " +
                  "the geographic range")
            print(f'{len(profiles_in_range_dataframe)} profiles associated with those floats')
        return profiles_in_range


    def __get_in_date_range(self, dataframe_to_filter: pd)-> list:
        """ A function to create and return a true false array indicating
            profiles that fall within the date range.
        """
        # If filtering by floats has resulted in an empty dataframe being passed
        if dataframe_to_filter.empty:
            return [True] * len(dataframe_to_filter)
        # If the user has passed us the entire available date don't go through the whole
        # process of checking if the points of all the floats are inside the range
        beginning_of_full_range = np.datetime64(datetime(1995, 1, 1, tzinfo=timezone.utc))
        end_of_full_range = np.datetime64(datetime.now(timezone.utc))
        if self.start_date == beginning_of_full_range and self.end_date >= end_of_full_range:
            return [True] * len(dataframe_to_filter)
        if self.download_settings.verbose:
            print('Sorting floats for those within the date range...')
        # Define a t/f array for dates within the range
        profiles_in_range  = ((dataframe_to_filter['date'] > self.start_date) &
                              (dataframe_to_filter['date'] < self.end_date)).tolist()
        if self.download_settings.verbose:
            profiles_in_range_dataframe = dataframe_to_filter[profiles_in_range]
            print(f"{len(profiles_in_range_dataframe['wmoid'].unique())} floats fall within " +
                  'the date range')
            print(f'{len(profiles_in_range_dataframe)} profiles associated with those floats')
        return profiles_in_range


    def __get_in_time_and_space_constraints(self, dataframe_to_filter: pd)-> pd:
        """ A function to apply the 'outside' kwarg constraints to the results after filtering by
            space and time.
        """
        # Generate t/f arrays for profiles according to geographic and date range
        profiles_in_space = self.__get_in_geographic_range(dataframe_to_filter)
        profiles_in_time = self.__get_in_date_range(dataframe_to_filter)
        # Converting to np arrays so we can combine to make constraints
        profiles_in_space = np.array(profiles_in_space, dtype=bool)
        profiles_in_time = np.array(profiles_in_time, dtype=bool)
        constraints = profiles_in_time & profiles_in_space
        floats_in_time_and_space = dataframe_to_filter[constraints]
        floats_in_time_and_space = \
            np.array(dataframe_to_filter['wmoid'].isin(floats_in_time_and_space['wmoid']),
                     dtype=bool)
        # Filter passed dataframe by time and space constraints to
        # create a new dataframe to return as part of the selection frame
        if self.outside == 'time':
            if self.download_settings.verbose:
                print(f'Applying outside={self.outside} constraints...')
            constraints = floats_in_time_and_space & profiles_in_space
            selection_frame = dataframe_to_filter[constraints]
        elif self.outside == 'space':
            if self.download_settings.verbose:
                print(f'Applying outside={self.outside} constraints...')
            constraints = floats_in_time_and_space & profiles_in_time
            selection_frame = dataframe_to_filter[constraints]
        elif self.outside is None:
            if self.download_settings.verbose:
                print('Applying outside=None constraints...')
            constraints = floats_in_time_and_space & profiles_in_space & profiles_in_time
            selection_frame = dataframe_to_filter[constraints]
        elif self.outside == 'both':
            if self.download_settings.verbose:
                print(f'Applying outside={self.outside} constraints...')
            constraints = floats_in_time_and_space
            selection_frame = dataframe_to_filter[constraints]
        return selection_frame


    def __get_in_ocean_basin(self):
        """ A function to drop floats/profiles outside of the specified ocean basin.
        """
        if self.download_settings.verbose:
            print("Sorting floats for those passed in 'ocean' kwarg...")
        self.selection_frame = self.selection_frame[self.selection_frame['ocean'] ==
                                                    str(self.ocean)]
        if self.download_settings.verbose:
            print(f"{len(self.selection_frame['wmoid'].unique())} floats fall within " +
                  'the ocean basin')
            print(f'{len(self.selection_frame)} profiles fall within the ocean basin')


    def __dataframe_to_dictionary(self)-> dict:
        """ A function to turn the working directory into a dictionary
            of float keys with a list of profiles that match the criteria.
            :return: narrowed_profiles : dict - A dictionary with float ID
                keys corresponding to a list of profiles that match criteria.
        """
        selected_profiles = {}
        for index, row in self.selection_frame.iterrows():
            if row['wmoid'] not in selected_profiles:
                selected_profiles[row['wmoid']] = [row['profile_index']]
            else:
                selected_profiles[row['wmoid']].append(row['profile_index'])
        # Sort dict by key values
        float_ids = list(selected_profiles.keys())
        float_ids.sort()
        selected_profiles = {i: selected_profiles[i] for i in float_ids}
        return selected_profiles


    def __filter_by_floats(self)-> pd:
        """ Function to pull profiles of floats passed to trajectories() and return
            a dataframe with floats from sprof and prof index frames.
            :returns: floats_profiles: pd - The dataframe with only the profiles of
                the passed floats.
        """
        # Gather bgc profiles for these floats from sprof index frame
        bgc_filter = ((self.float_stats['wmoid'].isin(self.float_ids)) &
                      (self.float_stats['is_bgc'] == True))
        floats_bgc = self.float_stats.loc[bgc_filter, 'wmoid'].tolist()
        floats_bgc = self.sprof_index[self.sprof_index['wmoid'].isin(floats_bgc)]
        # Gather phys profiles for these floats from prof index frame
        phys_filter = ((self.float_stats['wmoid'].isin(self.float_ids)) &
                       (self.float_stats['is_bgc'] == False))
        floats_phys = self.float_stats[phys_filter]['wmoid'].tolist()
        floats_phys = self.prof_index[self.prof_index['wmoid'].isin(floats_phys)]
        # If the user has passed a dictionary also filter by profiles
        if self.float_profiles_dict is not None:
            # Flatten the float_dictionary into a DataFrame
            data = []
            for wmoid, profile_indexes in self.float_profiles_dict.items():
                if len(profile_indexes) == 1:
                    # If there is only one profile index, add it directly
                    data.append({'wmoid': wmoid, 'profile_index': profile_indexes[0]})
                else:
                    # Calculate the differences between consecutive elements
                    nans_needed = np.diff(profile_indexes)
                    # Add elements and nans
                    for i in range(1, len(profile_indexes)):
                        # If the difference is greater than 1, insert NaNs
                        if nans_needed[i-1] > 1:
                            data.append({'wmoid': wmoid, 'profile_index': np.nan})
                        # Add the current profile index
                        data.append({'wmoid': wmoid, 'profile_index': profile_indexes[i]})
            # Convert the list of dictionaries into a DataFrame
            profile_df = pd.DataFrame(data)
            # Filter only profiles included in dataframe for bgc floats
            floats_bgc = pd.merge(floats_bgc, profile_df, on=['wmoid', 'profile_index'],
                                  how='right')
            floats_bgc = floats_bgc.reset_index(drop=True)
            # Filter only profiles included in the dataframe for phys floats
            floats_phys = pd.merge(floats_phys, profile_df, on=['wmoid', 'profile_index'],
                                   how='right')
            floats_phys = floats_phys.reset_index(drop=True)
        floats_profiles = pd.concat([floats_bgc, floats_phys])
        return floats_profiles


    def __set_graph_limits(self, ax, axis: str)-> None:
        """ A Function for setting the graph's longitude and latitude extents.
        """
        if axis == 'x':
            minimum, maximum = ax.get_xlim()
            diff = maximum - minimum
        elif axis == 'y':
            minimum, maximum = ax.get_ylim()
            diff = maximum - minimum
        if diff < 5.0:
            # Add padding to get at least 5 degrees of longitude
            pad = 0.5 * (5.0 - diff)
            minimum -= pad
            maximum += pad
            if axis == 'x':
                ax.set_xlim([minimum, maximum])
            elif axis == 'y':
                ax.set_ylim([minimum, maximum])


    def __add_grid_lines(self, ax)-> None:
        """ Function for setting the gridlines of passed graph.
        """
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.ylines = True
        step_x = self.__determine_graph_step(ax, 'x')
        step_y = self.__determine_graph_step(ax, 'y')
        longitude_ticks = list(range(-180, 181, step_x))
        latitude_ticks = list(range(-90, 91, step_y))
        gl.xlocator = FixedLocator(longitude_ticks)
        gl.ylocator = FixedLocator(latitude_ticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}


    def __determine_graph_step(self, ax, axis: str)-> int:
        """ A graph to determine the step of the longitude and latitude gridlines.
        """
        if axis == 'x':
            minimum, maximum = ax.get_xlim()
            diff = maximum - minimum
        elif axis == 'y':
            minimum, maximum = ax.get_ylim()
            diff = maximum - minimum
        if diff > 80:
            step = 15
        elif diff > 30:
            step = 10
        elif diff > 15:
            step = 5
        else:
            step = 2
        return step


    def __variable_permutations(self, nc_file)-> list:
        """ A function to filter the list of variables to be loaded so
            that we only load variables that are in the file.
            :param: nc_file : Any - The .nc file we're reading from.
            :return: list - A list to of all the variables passed
                that are inside of the nc_file.
        """
        # If the variables is in the file also add it's permutations to the list
        if isinstance(self.float_variables, list):
            # Parameters that are in the passed .nc file
            file_variables = nc_file.variables
            # List to store variables and their additioal associated columns
            variable_columns = []
            for variable in self.float_variables:
                if variable in file_variables:
                    # We add PRES no matter what, so if the user passed it
                    # don't add it to the variable list at this time.
                    if variable != 'PRES':
                        variable_columns.append(variable)
                        variable_columns.append(variable + '_QC')
                        variable_columns.append(variable + '_ADJUSTED')
                        variable_columns.append(variable + '_ADJUSTED_QC')
                        variable_columns.append(variable + '_ADJUSTED_ERROR')
                else:
                    print(f'WARNING: {variable} does not exist in File {nc_file.filepath()}.')
            if len(variable_columns) > 0:
                pressure = ['PRES', 'PRES_QC', 'PRES_ADJUSTED', 'PRES_ADJUSTED_QC',
                            'PRES_ADJUSTED_ERROR']
                existing_variable_columns = pressure + variable_columns
                return existing_variable_columns
        return None


    def __fill_float_data_dataframe(self, files: list)-> pd:
        """ A Function to load data into the float data dataframe.
            :param: files : list - A list of files to read in data from.
            :return: pd : Dataframe - The dataframe of float data with rows
                where measurements were not collected excluded.
        """
        if self.download_settings.verbose:
            print('Loading float data...')
        # Getting the file paths for downloaded .nc files
        directory = Path(self.download_settings.base_dir.joinpath("Profiles"))
        file_paths = []
        for file in files:
            file_paths.append(directory.joinpath(file))
        # Columns that will always be in the dataframe, these columns are one dimensional
        static_columns = ['WMOID', 'CYCLE_NUMBER', 'DIRECTION',
                                'DATE', 'DATE_QC', 'LATITUDE',
                                'LONGITUDE', 'POSITION_QC']
        # Columns that need to be calculated or derived
        special_case_static_columns = ['DATE', 'DATE_QC', 'WMOID']
        # Dataframe to return at end of function with all loaded data added
        float_data_dataframe = pd.DataFrame()
        # Iterate through files
        for file in file_paths:
            # Open File
            nc_file = netCDF4.Dataset(file, mode='r')
            # Get dimensions of .nc file
            number_of_profiles = nc_file.dimensions['N_PROF'].size
            number_of_levels = nc_file.dimensions['N_LEVELS'].size
            # Get float id of current file
            float_id_array = nc_file.variables['PLATFORM_NUMBER'][0]
            float_id = int(float_id_array.data.tobytes().decode('utf-8').strip('\x00'))
            # Get the range of profiles from the index file
            # If the file ends in Sprof then use the sprof index for profile count
            if 'Sprof' in str(file):
                profile_count = self.sprof_index['wmoid'].value_counts().get(float_id, 0)
            # Else use the prof index for profile count
            else:
                profile_count = self.prof_index['wmoid'].value_counts().get(float_id, 0)
            # Load only passed profiles if requested (floats is a dictionary)
            if self.float_profiles_dict is not None:
                if profile_count > number_of_profiles:
                    if self.download_settings.verbose:
                        print(f'Skipping float {float_id}...')
                        print(f'The index file has {profile_count} profiles and the .nc file ' +
                              f'has {number_of_profiles} profiles for float {float_id}')
                    continue
                # Get list of profiles passed in dictionary for float
                profiles_to_pull = self.float_profiles_dict[float_id]
                # Adjusting profile to index correctly from .nc file arrays
                profiles_to_pull = [index - 1 for index in profiles_to_pull]
                # The case for if we are pulling a single profile
                if len(profiles_to_pull) == 1:
                    profiles_to_pull = int(profiles_to_pull[0])
                    static_length = 1
                else:
                    static_length = len(profiles_to_pull)
            # If no profiles are passed then we want to pull all of the profiles from the float
            else:
                profiles_to_pull = list(range(0, number_of_profiles, 1))
                static_length = number_of_profiles
            # Narrow variable list to only thoes that are in the current file
            variable_columns = self.__variable_permutations(nc_file)
            # Temporary dataframe to make indexing simpler for each float
            temp_frame = pd.DataFrame()
            if self.download_settings.verbose:
                print(f'Loading Float data from float {float_id} with {static_length} profiles...')
            # Iterate through static columns
            for column in static_columns:
                # Customize nc_variable if we have a special case where values need to be calculated
                if column in special_case_static_columns:
                    nc_variable = self.__calculate_nc_variable_values(column, nc_file,
                                                                      static_length,
                                                                      profiles_to_pull)
                else:
                    nc_variable = nc_file.variables[column][profiles_to_pull]
                # Read in variable from .nc file
                column_values = self.__read_from_static_nc_variable(variable_columns, nc_variable,
                                                                    number_of_levels, static_length)
                if column.endswith('_QC'):
                    # Replace b'n' and b' ' with b'0' so that all values are numbers
                    modified_column = [b'0' if item in (b'n', b' ', b'') else item
                                       for item in column_values]
                    # These columns (DATE_QC and POSITION_QC) are always present, convert to int8
                    column_values = np.char.decode(modified_column, 'utf-8').astype('int8')
                if column == 'DIRECTION':
                    # The DIRECTION column is always present, convert to char
                    column_values = np.char.decode(column_values, 'utf-8')
                # Add list of values gathered for column to the temp dataframe
                temp_frame[column] = column_values
            # Iterate through variable columns, if there are none nothing happens
            if variable_columns is not None:
                for column in variable_columns:
                    # Setting nc_variable
                    nc_variable = nc_file.variables[column][profiles_to_pull,:]
                    # Replacing missing variables with NaNs
                    nc_variable = nc_variable.filled(np.nan)
                    # Read in variable from .nc file
                    column_values = self.__read_from_paramater_nc_variable(nc_variable)
                    if column.endswith('_QC'):
                        # Replace b'n' and b' ' with b'0' so that all values are numbers
                        modified_column = [b'0' if item in (b'n', b' ', b'') else item
                                           for item in column_values]
                        # Floats that do not have this column will have NaN here; convert to float
                        column_values = np.char.decode(modified_column, 'utf-8').astype('float')
                        # Add list of values gathered for column to the temp dataframe
                    temp_frame[column] = column_values
            # Clean up dataframe
            if 'PRES' in temp_frame.columns:
                if self.download_settings.verbose:
                    print(f'Dropping rows where no measurements were taken for {float_id}...')
                temp_frame = temp_frame.dropna(subset=['PRES', 'PRES_ADJUSTED'])
            # Concatonate the final dataframe and the temp dataframe
            float_data_dataframe = pd.concat([float_data_dataframe, temp_frame], ignore_index=True)
            # Close File
            nc_file.close()
        # Return dataframe
        return float_data_dataframe


    def __calculate_nc_variable_values(self, column: str, nc_file, number_of_profiles: int,
                                       profiles_to_pull: list) -> list:
        """ Function for specalized columns that must be calculated or derived.
            :param: files : list - A list of files to read in data from.
            :param: column : str - The name of the column of the dataframe we want information.
            :param: nc_file - The NC file object to read from.
            :param: number_of_profiles : int - The number of profiles expected to be read in.
            :param: profiles_to_pull : The indexes of the profiles we're pulling form the NC file.
            :return: list - The nc_variable adjusted for special cases.
        """
        if column == 'DATE' :
            # Acessing nc variables that we calculate date from
            nc_variable = nc_file.variables['JULD'][profiles_to_pull]
            # Making a list to store the calculated dates
            new_nc_variable = []
            # Check if nc_variable is 0-dimensional aka only one profile is passed
            if getattr(nc_variable, "shape", None) == ():
                reference_date = datetime(1950, 1, 1)
                utc_date = reference_date + timedelta(days=float(nc_variable))
                new_nc_variable.append(utc_date)
            else:
                # Calculating the dates
                for date in nc_variable:
                    reference_date = datetime(1950, 1, 1)
                    utc_date = reference_date + timedelta(days=date)
                    new_nc_variable.append(utc_date)
            # Returning list of calculated lists to be added to dataframe
            return new_nc_variable
        if column == 'DATE_QC':
            # Acessing nc variable that we pull date_qc from
            nc_variable = nc_file.variables['JULD_QC'][profiles_to_pull]
            # Returning nc variable
            return nc_variable
        if column == 'WMOID':
            # Parsing float id from file name
            float_id_array = nc_file.variables['PLATFORM_NUMBER'][0]
            float_id = int(float_id_array.data.tobytes().decode('utf-8').strip('\x00'))
            # List with the float id the same length as a one dimensional variable
            nc_variable = [int(float_id)] * number_of_profiles
            # Returning nc variable
            return nc_variable
        # this line should not be reached
        raise ValueError(f'Unexpected column name: {column}')


    def __read_from_static_nc_variable(self, variable_columns: list, nc_variable,
                                       number_of_levels: int, number_of_profiles: int)-> list:
        """ A function to read in data from one dimentional variables in the passed .nc file.
            :param: variable_columns : list - The list of variable columns in the .nc file. This
                determines how many times the static variables should be repeated to match the
                expected length of the dataframe.
            :param: nc_variable - The variable we're reading from.
            :param: number_of_levels : int - The number of depth levels per profile.
            :param: number_of_profiles : int - The number of profiles being pulled from a float.
            :return: list - The list of values for that nc_variable.
        """
        column_values = []
        # Check if nc_variable is 0-dimensional aka only one profile is passed
        if getattr(nc_variable, "shape", None) == ():
            column_values = [nc_variable] * (number_of_levels if variable_columns
                                             else number_of_profiles)
        # If there are no variables then we'll only need the rows to match the number of
        # profiles in the file
        elif variable_columns is None:
            for value in nc_variable:
                column_values.append(value)
        # If there are variables then the static rows need to match the number of levels
        else:
            for value in nc_variable:
                value_repeats = [value] * number_of_levels
                column_values.extend(value_repeats)
        return column_values


    def __read_from_paramater_nc_variable(self, nc_variable)-> list:
        """ A function to read in data from two dimentional variables in the passed .nc file.
            :param: nc_variable - The nc variable we're reading from
            :return: list - A list of values pulled from the nc variable passed.
        """
        column_values = []
        # Check if nc_variable is 0-dimensional aka only one profile is passed
        if nc_variable.ndim == 1:
            for profile in nc_variable:
                column_values.append(profile)
        else:
            for profile in nc_variable:
                for depth in profile:
                    column_values.append(depth)
        return column_values


    def __plot_section(self, all_float_data, float_id, variable, visible, save_to)-> None:
        """ A function to create a single section plot
            using the passed dataframe and variable.
        """
        # Grid Data
        float_data = all_float_data[all_float_data['WMOID'] == float_id]
        time_grid, pres_grid, param_gridded = self.__grid_section_data(float_data, variable)
        # Plot Data
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(time_grid, pres_grid, param_gridded, shading='auto')
        # Y Axis
        plt.ylim([0, float_data['PRES'].max()])
        plt.gca().invert_yaxis()
        # Add a colorbar to show the scale of the variable
        plt.colorbar(label=variable)
        # X Axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        # Titles
        plt.xlabel('Time')
        plt.ylabel('Pressure (dbar)')
        plt.title(f'{variable} Section of Float {float_id}')
        # Saving Graph
        if save_to is not None:
            save_path = save_to.joinpath(f'section_{float_id}_{variable}')
            plt.savefig(f'{save_path}')
        # Displaying graph
        if visible:
            plt.show()


    def __grid_section_data(self, float_data, variable):
        """ Function to grid the data
        """
        # Parse out values for specified float
        time_values = pd.to_datetime(float_data['DATE']).values
        pres_values = float_data['PRES'].values
        param_values = float_data[variable].values
        # Remove NaN values
        valid_indices = ~np.isnan(time_values) & ~np.isnan(pres_values) & ~np.isnan(param_values)
        time_values = time_values[valid_indices]
        pres_values = pres_values[valid_indices]
        param_values = param_values[valid_indices]
        # Convert time_values to float because it makes gridding data easier
        time_values_num = mdates.date2num(time_values)
        # Unique values for creating grids
        unique_times_num = np.unique(time_values_num)
        # Create a pressure axis with regular intervals, covering all existing values
        intp_pres = np.arange(np.ceil(min(pres_values)), np.floor(max(pres_values)))
        # Create grid for interpolation
        time_grid, pres_grid = np.meshgrid(unique_times_num, intp_pres)
        # Set param_gridded to NaN array with the same shape as the grid
        param_gridded = np.full(time_grid.shape, np.nan)
        # Create a DataFrame
        df = pd.DataFrame({
            'time': time_values_num,
            'pressure': pres_values,
            'param': param_values
        })
        # Pivot the DataFrame to create a grid
        param_gridded_df = df.pivot_table(
            index='pressure',
            columns='time',
            values='param',
            aggfunc='first'
        )
        # Create a new index that contains original and regularly spaced pressure values
        all_pres = np.sort(np.unique(np.concatenate([pres_values, intp_pres])))
        # Reindex the DataFrame to the combined depth axis
        param_gridded_df = param_gridded_df.reindex(index=all_pres, columns=unique_times_num)
        # Perform linear interpolation to the new depth axis without extrapolation
        param_gridded_df.interpolate(method='linear', limit_area='inside', axis=0, inplace=True)
        # Extract the values to the regularly spaced depth values
        param_gridded_df = param_gridded_df.reindex(index=intp_pres, columns=unique_times_num)
        # Assigning data to variable to graph
        param_gridded = param_gridded_df.values
        return time_grid, pres_grid, param_gridded
