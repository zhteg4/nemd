from datetime import datetime

Xx_FMT = '%X %x'
HMS_FMT = '%H:%M:%S'
HMS_ZERO = datetime.strptime('00:00:00', HMS_FMT)


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


def delta2str(delta, fmt=HMS_FMT):
    """
    Convert a timedelta object to a string representation.

    :param delta: the timedelta object to convert
    :type delta: 'datetime.timedelta'
    :param fmt: the format to print the time
    :type fmt: str
    :return: the string representation of the timedelta object
    :rtype: str
    """
    return (HMS_ZERO + delta).strftime(fmt)


def str2delta(value, fmt=HMS_FMT):
    """
    Convert a string representation of time to a timedelta object.

    :param value: the string representation of time
    :type value: str
    :param fmt: the format to parse the input string
    :type fmt: str
    :return: the timedelta object based on input string
    :rtype: 'datetime.timedelta'
    """
    hms = dtime(value, fmt=fmt)
    return hms - HMS_ZERO
