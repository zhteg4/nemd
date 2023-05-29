from datetime import datetime

Xx_FMT = '%X %x'


def ctime(fmt=Xx_FMT):
    """
    Get current time.

    :param fmt: the format to print current time
    :type fmt: str
    :return: current time
    :rtype: str
    """

    return datetime.now().strftime(fmt)


def dtime(strftime, fmt=Xx_FMT):
    """
    Get the datatime from str time.

    :param strftime:
    :type strftime: str
    :param fmt: the format to parse input time str
    :type fmt: str
    :return: the datatime based on input str
    :rtype: 'datetime.datetime'
    """

    return datetime.strptime(strftime, fmt)
