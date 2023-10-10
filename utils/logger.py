import os
import json
import time
import platform
import logging

def logging_config(args, process_id):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    console = logging.StreamHandler()
    if args.level == 'INFO':
        console.setLevel(logging.INFO)
    elif args.level == 'DEBUG':
        console.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    formatter = logging.Formatter(str(process_id) + 
        ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)
    # logging.getLogger().info("test")
    logging.basicConfig()
    logger = logging.getLogger()
    if args.level == 'INFO':
        logger.setLevel(logging.INFO)
    elif args.level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    logging.info(args)



class Logger(object):

    INFO = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @classmethod
    def config_logger(cls, file_folder='.', level="info",
                        save_log=False, display_source=False):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        cls.file_folder = file_folder
        cls.file_json = os.path.join(file_folder, "log-1.json")
        # cls.file_log can be changed by add_log_file()
        cls.file_log = os.path.join(file_folder, "log.log")
        cls.values = []
        cls.save_log = save_log
        logger = logging.getLogger()
        if display_source:
            cls.formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
        else:
            cls.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        cls.level = level
        if level == "info":
            logger.setLevel(logging.INFO)
        elif level == "debug":
            logger.setLevel(logging.DEBUG)
        elif level == "warning":
            logger.setLevel(logging.WARNING)
        elif level == "error":
            logger.setLevel(logging.ERROR)
        elif level == "critical":
            logger.setLevel(logging.CRITICAL)

        strhdlr = logging.StreamHandler()
        strhdlr.setFormatter(cls.formatter)
        logger.addHandler(strhdlr)
        if save_log:
            cls.add_log_file(cls.file_log)
        cls.logger = logger


    @classmethod
    def add_log_file(cls, logfile):
        assert cls.save_log is True
        hdlr = logging.FileHandler(logfile)
        hdlr.setFormatter(cls.formatter)
        cls.logger.addHandler(hdlr) 


    @classmethod
    def display_metric(cls, name, values, tags):
        cls.info(
            value="{name} ({tags}): {values} ".format(
                name=name, values=values)
        )


    @classmethod
    def cache_metric_in_memory(cls, name, values, tags):
        """
        Store a scalar metric. Example:
        name="runtime",
        values={
            "time": current_time,
            "rank": rank,
            "epoch": epoch,
            "best_perf": best_perf,
        },
        tags={"split": "test", "type": "local_model_avg"},
        """
        cls.values.append({"measurement": name, **tags, **values})


    @classmethod
    def log_timer(cls, name, values, tags):
        cls.info(
            value="{name} ({tags}): {values} ".format(
                name=name, values=values)
        )


    @classmethod
    def info(cls, value):
        cls.logger.info(value)

    @classmethod
    def debug(cls, value):
        cls.logger.debug(value)

    @classmethod
    def warning(cls, value):
        cls.logger.warning(value)
    
    @classmethod
    def error(cls, value):
        cls.logger.error(value)

    @classmethod
    def critical(cls, value):
        cls.logger.critical(value)


    @classmethod
    def save_json(cls):
        """Save the internal memory to a file."""
        with open(cls.file_json, "w") as fp:
            json.dump(cls.values, fp, indent=" ")

        if len(cls.values) > 1e3:
            # reset 'values' and redirect the json file to other name.
            cls.values = []
            cls.redirect_new_json()


    @classmethod
    def redirect_new_json(cls):
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(cls.file_folder) if "json" in file
        ]
        cls.file_json = os.path.join(
            cls.file_folder, "log-{}.json".format(len(existing_json_files) + 1)
        )


# Usage example
def display_training_stat(conf, tracker, epoch, n_bits_to_transmit):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # display the runtime training information.
    Logger.display_metric(
        name="runtime",
        values={
            "time": current_time,
            "epoch": epoch,
            "n_bits_to_transmit": n_bits_to_transmit / 8 / (2 ** 20),
            **tracker(),
        },
        tags={"split": "train"}
    )


# Usage example
def display_test_stat(conf, tracker, epoch, label="local"):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # display the runtime training information.
    Logger.display_metric(
        name="runtime",
        values={
            "time": current_time,
            "epoch": epoch,
            **tracker(),
        },
        tags={"split": "test", "type": label}
    )
