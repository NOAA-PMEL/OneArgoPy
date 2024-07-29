# -*- coding: utf-8 -*-
# OneArgoSettings.py
#------------------------------------------------------------------------------
# Created By: Savannah Stephenson and Hartmut Frenzel
# Creation Date: 07/26/2024
# Version: 0.1 (alpha)
#------------------------------------------------------------------------------
""" This file holds the classes that the library will use to describe 
    various settings involved in the process of extracting and 
    plotting data from the Argo floats. All attributes will have a
    default value that is assigned upon use of the class that can be
    viewed and altered with setter and getter functions.  
"""
#------------------------------------------------------------------------------
#
#
# Imports

# System
from pathlib import Path
import json
import pandas as pd


class DownloadSettings():
    """ The DownloadSettings class is used to store all of the information
        needed in to create directories to store downloaded data from 
        the Global Data Assembly Center (GDAC), when to log downloads, and 
        when to update downloaded data.

        :param: user_settings : str - An optional path to user defined 
            settings.

        Settings:
        base_dir : str - The base directory that all sub directories 
            should be created at.
        sub_dirs : list - A list of folders to that will store 
            downloaded data.
        index_files : list - A list of the index files that will be 
            downloaded.
        verbose : bool - A boolean value that determines whether to 
            log verbosely or not.
        update : int - An integer value that determines the threshold
            for updating downloaded files (0: do not update; >0: maximum 
            number of seconds since an index file was downloaded before 
            downloading it again for new profile selection).
        max_attempts : int - An integer value that determines the 
            number of times the library tries to download the same file before 
            raising an exception.
        keep_index_in_memory : bool - True by default, a value to 
            determine if the dataframes from the index files should be 
            kept in working memory or not.
        float_type : str - 'all' by default, a string indicating the type
            of floats that the researcher would like to handle. Valid options
            are 'bgc', 'phys', and 'all'.
        timeout : int - An integer value representing the number of seconds
            to wait for a web server to respond to a download request;
            default value: 300 (5 minutes). 
    """
    def __init__(self, user_settings: str = None) -> None:
        if user_settings is not None:
            ds_data = self.__parse_download_settings(Path(user_settings))
            self.base_dir = Path(ds_data['base_dir'])
            self.sub_dirs = ds_data['sub_dirs']
            self.index_files = ds_data['index_files']
            self.verbose = ds_data['verbose']
            self.update = ds_data['update']
            self.max_attempts = ds_data['max_attempts']
            self.keep_index_in_memory = ds_data['keep_index_in_memory']
            self.float_type = ds_data['float_type']
            self.timeout = ds_data['timeout']
        else:
            self.base_dir =  Path(__file__).resolve().parent
            self.sub_dirs =  ["Index", "Meta", "Tech", "Traj", "Profiles"]
            self.index_files =  ["ar_index_global_traj.txt", "ar_index_global_tech.txt",
                                 "ar_index_global_meta.txt", "ar_index_global_prof.txt",
                                 "argo_synthetic-profile_index.txt"]
            self.verbose = True
            self.update = 3600
            self.max_attempts = 10
            self.keep_index_in_memory = True
            self.float_type = "all"
            self.timeout = 300


    def __parse_download_settings(self, user_settings: Path) -> dict:
        """ A function to parse a given user_settings file to initialize
            the DownloadSettings class based off of a passed path to a json
            file. 

            :param: user_settings : Path - The path to the user's settings file

            :returns: ds_data : dict - The parsed json string to assign to DownloadSettings
                parameters. 
        """
        if not user_settings.exists():
            print(f'{user_settings} not found!')
            raise FileNotFoundError

        with user_settings.open('r', encoding='utf-8') as file:
            data = json.load(file)

        # Parse DownloadSettings
        ds_data = data['DownloadSettings']
        return ds_data


    def __str__(self) -> str:
        return (f'\n[DownloadSettings] -> \nBase Directory: {self.base_dir}, ' +
                f'\nSubdirectories: {self.sub_dirs}, \nIndex Files: {self.index_files}, ' +
                f'\nVerbose Setting: {self.verbose}, \nMax Attempts: {self.max_attempts}, ' +
                f'\nKeep Index In Memory: {self.keep_index_in_memory}, ' +
                f'\nFloat Type: {self.float_type}\n')


    def __repr__(self) -> str:
        return '\nDownloadSettings(path_to_user_settings_file)'


    def __eq__(self, __value: object) -> bool:
        return (self.base_dir == __value.base_dir and
            self.sub_dirs == __value.sub_dirs and
            self.index_files == __value.index_files and
            self.verbose == __value.verbose and
            self.update == __value.update and
            self.max_attempts == __value.max_attempts and
            self.keep_index_in_memory == __value.keep_index_in_memory and
            self.float_type == __value.float_type)


class AnalysisSettings():
    """ The AnalysisSettings class is used to store all of the default 
        settings for analyzing data from the Argo floats.

        :param: user_settings : str - An optional path to user defined 
            settings.

        Settings: 
        temp_thresh : float - The temperature threshold for mixed 
            layer depth calculations measured in degrees Celsius. 
        dens_thresh : float - The density threshold for mixed layer 
            depth calculations measured in kg/m^3.
        interp_lonlat : bool - A boolean value determining whether 
            or not to interpolate missing latitude and longitude values
    """
    def __init__(self, user_settings: str = None) -> None:
        if user_settings is not None:
            as_data = self.__parse_analysis_settings(Path(user_settings))
            self.temp_thresh=as_data['temp_thresh']
            self.dens_thresh=as_data['dens_thresh']
            self.interp_lonlat=as_data['interp_lonlat']
        else:
            self.temp_thresh = 0.2
            self.dens_thresh = 0.03
            self.interp_lonlat = False


    def __parse_analysis_settings(self, user_settings: Path) -> dict:
        """ A function to parse a given user_settings file to initialize
            the AnalysisSettings class based off of a passed path to a json
            file. 

            :param: user_settings : Path - The path to the user's settings file

            :returns: ds_data : dict - The parsed json string to assign to AnalysisSettings
                parameters. 
        """
        if not user_settings.exists():
            print(f'{user_settings} not found!')
            raise FileNotFoundError

        with user_settings.open('r', encoding='utf-8') as file:
            data = json.load(file)

        # Parse DownloadSettings
        as_data = data['AnalysisSettings']
        return as_data


    def __str__(self) -> str:
        return (f'\n[Analysis Settings] -> \nTemperature Threshold: {self.temp_thresh}, ' +
                f'\nDensity Threshold: {self.dens_thresh}, ' +
                f'\nInterpolate Latitude and Longitude: {self.interp_lonlat}\n')


    def __repr__(self) -> str:
        return '\nAnalysisSettings(path_to_user_settings_file)'


    def __eq__(self, __value: object) -> bool:
        return (self.temp_thresh == __value.temp_thresh and
            self.dens_thresh == __value.dens_thresh and
            self.interp_lonlat == __value.interp_lonlat)


class SourceSettings():
    """ The SourceSettings class is used to store information about where 
        we are collecting the Argo Float data from.

        :param: user_settings : str - An optional path to user defined 
            settings.

        Settings:
        hosts : list - The US and French GDAC URLs. IFREMER is often
            faster than GODAE so it is listed first.
        avail_vars : list - The full set of available variables, 
            will be filled during evaluation of the index files.
        dacs : list - A list of Data Assimilation Centers, will be 
            filled during evaluation of the index files. 
    """
    def __init__(self, user_settings: str = None) -> None:
        if user_settings is not None:
            ss_data = self.__parse_source_settings(Path(user_settings))
            self.hosts = ss_data['hosts']
            self.avail_vars = ss_data['avail_vars']
            self.dacs = ss_data['dacs']
        else:
            self.hosts = ["https://data-argo.ifremer.fr/",
                          "https://usgodae.org/ftp/outgoing/argo/"]
            self.avail_vars = None
            self.dacs = None


    def __parse_source_settings(self, user_settings: Path) -> dict:
        """ A function to parse a given user_settings file to initialize
            the SourceSettings class based off of a passed path to a json
            file. 

            :param: user_settings : Path - The path to the user's settings file

            :returns: ds_data : dict - The parsed json string to assign to SourceSettings
                parameters. 
        """
        if not user_settings.exists():
            print(f'{user_settings} not found!')
            raise FileNotFoundError

        with user_settings.open('r', encoding='utf-8') as file:
            data = json.load(file)

        # Parse DownloadSettings
        ss_data = data['SourceSettings']
        return ss_data


    def set_avail_vars(self, synthetic_index: pd) -> None:
        """ A function to dynamically fill the avail_vars parameter from the
            source settings with variables from the argo_synthetic_profile_index.
        """
        all_parameters = synthetic_index['parameters'].str.split().explode()
        unique_parameters = all_parameters.unique()
        self.avail_vars = unique_parameters.tolist()


    def set_dacs(self, synthetic_index: pd) -> None:
        """ A function to dynamically fill the dacs parameter from the
            source settings with variables from the argo_synthetic_profile_index.
        """
        unique_dacs = synthetic_index['dacs'].unique()
        self.dacs = unique_dacs.tolist()


    def __str__(self) -> str:
        return (f'\n[Source Settings] -> \nHosts: {self.hosts}, ' +
                f'\nAvailable Variables: {self.avail_vars}, ' +
                f'\nData Assimilation Centers: {self.dacs}\n')


    def __repr__(self) -> str:
        return '\nSourceSettings(path_to_user_settings_file)'


    def __eq__(self, __value: object) -> bool:
        return (self.hosts == __value.hosts and
            self.avail_vars == __value.avail_vars and
            self.dacs == __value.dacs)
