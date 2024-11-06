# ------------------------------------------------
# MEDSLIK-II oil spill fate and transport model
# ------------------------------------------------
import shutil
import logging
import importlib.util
from datetime import datetime

# Import medslik modules
from WITOIL_iMagine.src.utils import Utils, Config, read_oilbase
from WITOIL_iMagine.src.download import *
from WITOIL_iMagine.src.preprocessing import PreProcessing
from WITOIL_iMagine.src.postprocessing import PostProcessing
from WITOIL_iMagine.src.plot import MedslikIIPlot

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler with overwrite mode ('w')
file_handler = logging.FileHandler("medslik_run.log", mode="w")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


class MedslikII:
    """
    This class embeds the MAIN code of medslik-II software.
    """

    def __init__(self, config: dict) -> None:
        """
        Class constructor given config file path.
        """
        self.config = config
        # Create experiment directories
        self.root_directory = os.path.join(
            self.config["simulation"]["experiment_path"], config["simulation"]["name"]
        )
        os.makedirs(self.root_directory, exist_ok=True)
        self.out_directory = os.path.join(self.root_directory, "out_files")
        os.makedirs(self.out_directory, exist_ok=True)
        self.out_figures = os.path.join(self.out_directory, "figures")
        os.makedirs(self.out_figures, exist_ok=True)
        self.xp_directory = os.path.join(self.root_directory, "xp_files")
        os.makedirs(self.xp_directory, exist_ok=True)
        spill_lat = np.array(self.config["simulation"]["spill_lat"])
        self.n_spill_points = np.shape(spill_lat)[0]
        # Domain of the simulation will be defined under what the user set in config files
        if config["input_files"]["set_domain"] == True:
            logger.info("User defined domain")
            lat_min, lat_max = (
                config["input_files"]["lat"][0],
                config["input_files"]["lat"][1],
            )
            lon_min, lon_max = (
                config["input_files"]["lon"][0],
                config["input_files"]["lon"][1],
            )
        #Domain is based on a delta degrees
        else:
            logger.info(f"Domain defined around simulation point, \
                using {config['input_files']['delta'][0]} degrees")
            latitude = config["simulation"]["spill_lat"][0]
            longitude = config["simulation"]["spill_lon"][0]

            lat_min, lat_max = (
                latitude - config["input_files"]["delta"][0],
                latitude + config["input_files"]["delta"][0],
            )
            lon_min, lon_max = (
                longitude - config["input_files"]["delta"][0],
                longitude + config["input_files"]["delta"][0],
            )
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.apply_aging_effects()
        self.initial_checking()

    def apply_aging_effects(self) -> None:
        """
        Consider aging when setting
        the simulation length and the start datetime.
        """
        oilspill = self.config["simulation"]
        age = oilspill["slick_age"]
        if self.n_spill_points > 1:
            oilspill["sim_length"] += age
            oilspill["start_datetime"] -= pd.Timedelta(hours=age)
            oilspill["spill_duration"] = 0
        self.config["simulation"] = oilspill

    def initial_checking(self):
        """
        Check if any issue might derive from configuration.
        """
        # checking if the coordinates are on land
        lat = self.config["simulation"]["spill_lat"]
        lon = self.config["simulation"]["spill_lon"]
        # function to check if the spill location was put into land
        coastline_path = self.config["input_files"]["dtm"]["coastline_path"]
        sea = Utils.check_land(lon, lat, coastline_path)
        if sea == 0:
            raise ValueError(
                "Your coordinates lie within land. Please check your values again"
            )
        # checking dates
        dt = Utils.validate_date(self.config["simulation"]["start_datetime"])
        logger.info("No major issues found on dates and oil spill coordinates")

        # checking if starting from area spill
        if self.config["input_files"]["shapefile"]["shape_path"]:
            shapefile_path = self.config["input_files"]["shapefile"]["shape_path"]
            # if simulation starts from shapefile, the volume will be disconsidered
            if os.path.exists(shapefile_path):
                logger.info(
                    f"Simulation initial conditions area spill are provided on \
                        {self.config['input_files']['shapefile']['shape_path']}. \
                        Spill rate from config files will not be considered"
                )
                volume = Utils.oil_volume_shapefile(self.config)

                # Correcting volume on the config object
                self.config["simulation"]["spill_rate"] = volume

    @staticmethod
    def data_download_medslik(
            config: dict, domain: list[float], root_directory: str
    ) -> None:
        """
        Download METOCE datasets.
        """
        lon_min, lon_max, lat_min, lat_max = domain
        copernicus_user = config["download"]["copernicus_user"]
        copernicus_pass = config["download"]["copernicus_password"]

        date = pd.to_datetime(config["simulation"]["start_datetime"])

        identifier = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)

        inidate = date - pd.Timedelta(hours=1)
        enddate = date + pd.Timedelta(hours=config["simulation"]["sim_length"] + 24)

        if 30.37 < np.mean([lat_min, lat_max]) < 45.7 and -17.25 < np.mean([lon_min, lon_max]) < 36:
            down = "local"
        else:
            down = "global"

        if config["download"]["download_curr"]:
            output_path = "WITOIL_iMagine/data/COPERNICUS/"
            output_name = output_path + "Copernicus{}_{}_{}_mdk.nc".format(
                "{}", identifier, config["simulation"]["name"]
            )

            logger.info("Downloading CMEMS currents")
            download_copernicus(
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                0,
                120,
                inidate,
                enddate,
                down,
                output_path=output_path,
                output_name=output_name,
                user=copernicus_user,
                password=copernicus_pass,
            )

            subprocess.run(
                [
                    f'cp {output_path}*{identifier}*{config["simulation"]["name"]}*.nc {root_directory}/oce_files/'
                ],
                shell=True,
            )
            subprocess.run(
                [f'rm {output_path}*{identifier}*{config["simulation"]["name"]}*.nc'],
                shell=True,
            )

        if config["download"]["download_wind"]:
            # ensuring .cdsapirc is created in the home directory
            write_cds(config["download"]["cds_token"])

            output_path = "WITOIL_iMagine/data/ERA5/"
            output_name = output_path + "era5_winds10_{}_{}_mdk.nc".format(
                identifier, config["simulation"]["name"]
            )

            logger.info("Downloading ERA5 reanalysis winds")
            get_era5(
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                inidate,
                enddate,
                output_path=output_path,
                output_name=output_name,
            )
            process_era5(output_path=output_path, output_name=output_name)

            subprocess.run(
                [
                    f'cp {output_path}*{identifier}*{config["simulation"]["name"]}*.nc {root_directory}/met_files/'
                ],
                shell=True,
            )
            subprocess.run(
                [f'rm {output_path}*{identifier}*{config["simulation"]["name"]}*.nc'],
                shell=True,
            )

    def run_preproc(config: dict, exp_folder: str, lon_min, lon_max, lat_min, lat_max, n_spill_points):
        """
        Run preprocessing.
        """

        domain = [lon_min, lon_max, lat_min, lat_max]
        preproc = PreProcessing(config=config, exp_folder=exp_folder, domain=domain)
        # Create folders
        preproc.create_directories()

        # download data if needed
        if config["download"]["download_data"] == True:
            MedslikII.data_download_medslik(config, domain, exp_folder)

        if config["run_options"]["preprocessing"]:

            if config["run_options"]["preprocessing_metoce"]:
                oce_path = config["input_files"]["metoce"]["oce_data_path"]
                met_path = config["input_files"]["metoce"]["met_data_path"]
                if oce_path == "":
                    oce_path = None
                if met_path == "":
                    met_path = None
                # create Medslik-II current file inputs
                preproc.process_currents(oce_path=f"{exp_folder}/oce_files/")
                # create Medslik-II wind file inputs
                preproc.process_winds(met_path=f"{exp_folder}/met_files/")

            if config["run_options"]["preprocessing_dtm"]:
                # use the same grid on currents to crop bathymetry
                preproc.common_grid()
                # create Medslik-II bathymetry file inputs
                preproc.process_bathymetry(
                    config["input_files"]["dtm"]["bathymetry_path"]
                )
                # create Medslik-II coastline file inputs
                preproc.process_coastline(
                    config["input_files"]["dtm"]["coastline_path"]
                )

            if config["input_files"]["shapefile"]["shape_path"]:
                shapefile_path = config["input_files"]["shapefile"]["shape_path"]
                if os.path.exists(shapefile_path):
                    # using an area spill identified on a shapefile to generate initial conditions
                    preproc.process_initial_shapefile()

            spill_dictionary = {}
            if n_spill_points > 1:
                logger.info(
                    f"Starting to write {n_spill_points} events of oil spill"
                )
                for i, dur in enumerate(config["spill_rate"]):
                    # obtaining the variables
                    spill_dictionary["simname"] = preproc.simname
                    spill_dictionary["dt_sim"] = config["simulation"][
                        "start_datetime"
                    ]
                    spill_dictionary["sim_length"] = preproc.sim_length
                    spill_dictionary["longitude"] = config["simulation"][
                        "spill_lon"
                    ][i]
                    spill_dictionary["latitude"] = config["simulation"][
                        "spill_lat"
                    ][i]
                    spill_dictionary["spill_duration"] = int(
                        config["simulation"]["spill_duration"][i]
                    )
                    spill_dictionary["spill_rate"] = config["simulation"][
                        "spill_rate"
                    ][i]
                    spill_dictionary["oil_api"] = config["simulation"]["oil"][i]
                    spill_dictionary["number_slick"] = 1
                    preproc.write_config_files(
                        spill_dictionary, separate_slicks=True, s_num=i
                    )

            else:
                logger.info("Writing single slick event")
                spill_dictionary["simname"] = preproc.simname
                spill_dictionary["dt_sim"] = config["simulation"]["start_datetime"]
                spill_dictionary["sim_length"] = int(preproc.sim_length)
                spill_dictionary["longitude"] = config["simulation"]["spill_lon"][
                    0
                ]
                spill_dictionary["latitude"] = config["simulation"]["spill_lat"][0]
                spill_dictionary["spill_duration"] = int(
                    config["simulation"]["spill_duration"][0]
                )
                spill_dictionary["spill_rate"] = config["simulation"][
                    "spill_rate"
                ][0]
                spill_dictionary["oil_api"] = config["simulation"]["oil"][0]
                preproc.write_config_files(spill_dictionary, separate_slicks=False)

            logger.info("Modfying medslik_II.for")
            preproc.process_medslik_memmory_array()
            logger.info("Medslik-II simulation parameters")
            if config["simulation"]["advanced_parameters"] == False:
                logger.info("Using custom advanced parameters")
                preproc.configuration_parameters()
            else:
                logger.info("Using standard parameters")

    def run_medslik_sim(self, simdir, simname, separate_slicks=False):

        # model directory. Could be changed, but will remain fixed for the time being.
        model_dir = "WITOIL_iMagine/src/model/"

        day = self.config["simulation"]["start_datetime"].day
        year = self.config["simulation"]["start_datetime"].year
        month = self.config["simulation"]["start_datetime"].month
        hour = self.config["simulation"]["start_datetime"].hour
        minute = self.config["simulation"]["start_datetime"].minute

        output_dir = f"{model_dir}OUT/MDK_SIM_{year}_{month:02d}_{day:02d}_{hour:02d}{minute:02d}_{simname}/."

        # removing old outputes just to be sure
        subprocess.run([f"rm -rf {output_dir}"], shell=True)

        if separate_slicks == False:
            # copy METOCEAN files to MEDSLIK-II installation
            subprocess.run(
                [f"cp {simdir}{simname}/oce_files/*.mrc {model_dir}RUN/TEMP/OCE/"],
                shell=True,
                check=True,
            )
            subprocess.run(
                [f"cp {simdir}{simname}/met_files/*.eri {model_dir}RUN/TEMP/MET/"],
                shell=True,
                check=True,
            )
            # copy bnc files
            subprocess.run(
                [f"cp {simdir}{simname}/bnc_files/* {model_dir}DTM_INP/"],
                shell=True,
                check=True,
            )
            # copy Extract and config files
            subprocess.run(
                [
                    f"cp {simdir}{simname}/xp_files/medslik_II.for {model_dir}RUN/MODEL_SRC/"
                ],
                shell=True,
                check=True,
            )
            subprocess.run(
                [f"cp {simdir}{simname}/xp_files/config2.txt {model_dir}RUN/"],
                shell=True,
                check=True,
            )
            subprocess.run(
                [f"cp {simdir}{simname}/xp_files/config1.txt {model_dir}RUN/"],
                shell=True,
                check=True,
            )
            # Compile and start running
            subprocess.run(
                [f"cd {model_dir}RUN/; sh MODEL_SRC/compile.sh; ./RUN.sh"],
                shell=True,
                check=True,
            )

        else:
            slicks = glob(f"{simdir}{simname}/xp_files/*/")
            for i in range(0, len(slicks)):
                subprocess.run(
                    [f"cp {simdir}{simname}/oce_files/*.mrc {model_dir}RUN/TEMP/OCE/"],
                    shell=True,
                )
                subprocess.run(
                    [f"cp {simdir}{simname}/met_files/*.eri {model_dir}RUN/TEMP/MET/"],
                    shell=True,
                )
                # copy bnc files
                subprocess.run(
                    [f"cp {simdir}{simname}/bnc_files/* {model_dir}DTM_INP/"],
                    shell=True,
                )
                # copy Extract and config files
                subprocess.run(
                    [
                        f"cp {simdir}{simname}/xp_files/medslik_II.for {model_dir}RUN/MODEL_SRC/"
                    ],
                    shell=True,
                )
                subprocess.run(
                    [f"cp {simdir}{simname}/xp_files/config2.txt {model_dir}RUN/"],
                    shell=True,
                )
                subprocess.run(
                    [
                        f"cp {simdir}{simname}/xp_files/slick{i + 1}/config1.txt {model_dir}RUN/"
                    ],
                    shell=True,
                )
                # Compile and start running
                subprocess.run(
                    [f"cd {model_dir}RUN/; sh MODEL_SRC/compile.sh; ./RUN.sh"],
                    shell=True,
                    check=True,
                )

        # Send files to case dir and remove temp files
        subprocess.run([f"cp -r {output_dir} {simdir}{simname}/out_files/"], shell=True)
        subprocess.run(
            [f"rm -rf {simdir}{simname}/out_files/MET {simdir}{simname}/out_files/OCE"],
            shell=True,
        )

def main_run(config_path=None):
    # Logging first info
    logger.info("Starting Medslik-II oil spill simulation")
    exec_start_time = datetime.datetime.now()
    logger.info(f"Execution starting time = {exec_start_time}")

    logger.info("Defining the main object")
    config = Config(config_path).config_dict

    main = MedslikII(config)

    shutil.copy(config_path, os.path.join(main.xp_directory, "config.toml"))

    # performing initial checking
    main.initial_checking()

# Run preprocessing
    logger.info("Starting pre processing ... ")
    MedslikII.run_preproc(
        main.config,
        main.root_directory,
        main.lon_min,
        main.lon_max,
        main.lat_min,
        main.lat_max,
        main.n_spill_points
    )
    logger.info("End of pre processing ...")

    # Run model
    if main.config["run_options"]["run_model"]:
        logger.info("Running Medslik-II simulation")
        main.run_medslik_sim(
            "WITOIL_iMagine/cases/", main.config["simulation"]["name"], separate_slicks=False
        )

    #performing postprocessing
    if main.config["run_options"]["postprocessing"]:
        multiple_slick = main.config["simulation"]["multiple_slick"]
        PostProcessing.create_concentration_dataset(
            lon_min=main.lon_min,
            lon_max=main.lon_max,
            lat_min=main.lat_min,
            lat_max=main.lat_max,
            filepath=main.out_directory,
            multiple_slick=multiple_slick,
        )

    #plotting the results
    if main.config["plot_options"]["plotting"]:
        mplot = MedslikIIPlot(main)
        mplot.plot_matplotlib(main.lon_min,main.lon_max,main.lat_min,main.lat_max)
        try:
            mplot.plot_mass_balance()
        except:
            pass

    # shutil.copy("WITOIL_iMagine/medslik_run.log", f"{main.out_directory}medslik_run.log")

    # if config_path is None:
    #     shutil.copy("WITOIL_iMagine/config.toml", f"{main.out_directory}config.toml")
