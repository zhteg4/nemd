import sys
import hello
import argparse
import parserutils
import numpy as np

FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'


def load_energy_file(energy_file):
    with open(energy_file, 'r') as file_energy:
        start_line_num = 1
        one_line = file_energy.readline()
        while not one_line.startswith('Step'):
            start_line_num += 1
            one_line = file_energy.readline()
        data_names = one_line.split()
        try:
            np.loadtxt(file_energy)
        except ValueError as err:
            err_str = str(err)
            end_line_num = int(err_str.split()[-1]) - 1

    data_formats = ('int', 'float', 'float', 'float')
    ene_dat = np.loadtxt(energy_file,
                         dtype={
                             'names': data_names,
                             'formats': data_formats
                         },
                         skiprows=start_line_num,
                         max_rows=end_line_num)

    return ene_dat


def load_temp_file(temp_file):
    with open(temp_file, 'r') as file_temp:
        step_nbin_nave = np.loadtxt(file_temp, skiprows=3, max_rows=1)
        nbin = int(step_nbin_nave[1])
        data = np.zeros((nbin, 4))
        data_num = 0
        while file_temp:
            data_num += 1
            data += np.array(np.loadtxt(file_temp, max_rows=nbin))
            if not file_temp.readline():
                return data / data_num


def get_parser():
    parser = parserutils.get_parser(
        description=
        'Calculate thermal conductivity using non-equilibrium molecular dynamics.'
    )
    parser.add_argument(FLAG_TEMP_FILE,
                        metavar=FLAG_TEMP_FILE.upper(),
                        help='')
    parser.add_argument(FlAG_ENEGER_FILE,
                        metavar=FlAG_ENEGER_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


def main(argv):

    options = validate_options(argv)
    energy_dat = load_energy_file(options.energy_file)
    temp_dat = load_temp_file(options.temp_file)


if __name__ == "__main__":
    main(sys.argv[1:])
