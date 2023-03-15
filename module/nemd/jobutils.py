FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_DEBUG = '-DEBUG'


def add_job_arguments(parser, arg_flags=None):
    if arg_flags is None:
        arg_flags = [FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_DEBUG]

    if FLAG_INTERACTIVE in arg_flags:
        parser.add_argument(FLAG_INTERACTIVE,
                            dest=FLAG_INTERACTIVE[1:].lower(),
                            action='store_true',
                            help='')
    if FLAG_JOBNAME in arg_flags:
        parser.add_argument(FLAG_JOBNAME,
                            dest=FLAG_JOBNAME[1:].lower(),
                            help='')
    if FLAG_DEBUG in arg_flags:
        parser.add_argument(FLAG_DEBUG,
                            action='store_true',
                            dest=FLAG_DEBUG[1:].lower(),
                            help='')
