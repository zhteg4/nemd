import sys
import hello
import argparse
import parserutils

FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'

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

    option = validate_options(argv)
    import ipdb;
    ipdb.set_trace()
    with open(option.energy_file, 'r') as energy_file:
        energy_file.readline()

        import ipdb;ipdb.set_trace()
        pass
    hello.world()


if __name__ == "__main__":
    main(sys.argv[1:])
