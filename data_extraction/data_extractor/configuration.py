import json
import logging
import os
import inspect

dict_format_config = {}
src_dir = ""
work_dir = ""


def get_config(key: str, sub_key: str = None):
    """
    get configuration value
    """
    if key not in dict_format_config:
        raise ValueError("Cannot find key (%s) from configuration" % key)
    value = dict_format_config[key]
    if sub_key is None:
        if len(str(value)) == 0:
            raise ValueError("Empty value string for key (%s)" % key)
        return value
    if sub_key not in value:
        raise ValueError("Cannot find key (%s-%s) from configuration" % (key, sub_key))
    value = value[sub_key]
    if len(str(value)) == 0:
        raise ValueError("Empty value string for key (%s)" % sub_key)
    return value


def init(config_file_name="data_extraction.config"):
    """
    :param config_file_name: configuration file name inside source directory, or configuration file absolute path
    """

    def set_src_dir():
        """
        get source directory
        """
        global src_dir
        src_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        print("Source dir: %s" % src_dir)

    def load_config_file():
        """
        load configuration file
        """
        global dict_format_config
        nonlocal config_file_name
        if not os.path.isfile(config_file_name):
            config_file_name = os.path.join(src_dir, config_file_name)
        if not os.path.isfile(config_file_name):
            raise ValueError("Cannot find configuration file")
        with open(config_file_name, "r") as f:
            dict_format_config = json.load(f)
        if dict_format_config is None:
            raise ValueError("Cannot open configuration file.")

    def verify_and_apply_configurations():
        """
        verify correctness of configurations
        """
        # logging level
        log_level = get_config("logging level").upper()
        if log_level not in ["DEBUG", "INFO", "WARNING"]:
            raise ValueError("Logging level setting (%s) is incorrect." % log_level)
        log_level = getattr(logging, log_level)

        # working dir
        global work_dir
        work_dir = get_config("working dir")
        os.chdir(work_dir)

        # log settings
        log_file_name = "data_extractor.logging"
        logging.basicConfig(level=log_level,
                            filename=log_file_name,
                            format='%(levelname)s: %(asctime)s %(message)s',
                            datefmt='%m/%d %I:%M:%S')
        logging.info("Configurations: %s" % dict_format_config)
        logging.info("Source directory: %s" % src_dir)
        logging.info("Working directory switched to %s" % work_dir)

        # check configuration for alphavantage extractor
        get_config("alphavantage", "api_key")
        get_config("alphavantage", "limit_per_min")
        get_config("alphavantage", "limit_per_day")

    set_src_dir()
    load_config_file()
    verify_and_apply_configurations()

