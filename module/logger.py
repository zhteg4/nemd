import logging


def createLogger(jobname, verbose=True):
    logger = logging.getLogger(jobname)
    hdlr = logging.FileHandler(jobname + '-driver.log')
    if verbose:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
    logger.addHandler(hdlr)
    return logger
