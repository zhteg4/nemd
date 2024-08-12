import os
import re
import filecmp
import datetime
from nemd import task
from nemd import symbols

FLAG_DIR = '-dir'


class CmdParser:
    """
    The class to parse a test file.
    """

    NAME = 'cmd'
    POUND = symbols.POUND
    CMD_BRACKET_RE = '.*?\(.*?\)'
    AND_RE = r'and\s+'
    NAME_BRACKET_RE = re.compile('(.*?)\([\'|"]?(.*?)[\'|"]?\)')

    def __init__(self, path):
        """
        :param path str: the path containing the file
        """
        self.pathname = os.path.join(path, self.NAME)
        self.comment = None
        self.args = None

    def run(self):
        """
        Main method to parse one integration test file.
        """
        self.setArgs()
        self.removeUnkArgs()

    def setArgs(self):
        """
        Set arguments by reading the file.
        """
        if not os.path.isfile(self.pathname):
            return
        with open(self.pathname) as fh:
            self.args = [x.strip() for x in fh.readlines() if x.strip()]

    def removeUnkArgs(self):
        """
        Remove unknown arguments.
        """
        if self.args is None:
            return
        comments = [x for x in self.args if x.startswith(self.POUND)]
        comment = [x.strip(self.POUND).strip() for x in comments]
        self.comment = symbols.SPACE.join(comment)
        self.args = [x for x in self.args if not x.startswith(self.POUND)]


class CheckParser(CmdParser):

    NAME = 'check'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = []

    def run(self):
        """
        Parse the file and set the operators.
        """
        super().run()
        self.setOperators()

    def setOperators(self):
        """
        Parse the one line command to get the operators.
        """
        if self.args is None:
            return
        for match in re.finditer(self.NAME_BRACKET_RE, ' '.join(self.args)):
            name, value = [x.strip("'\"") for x in match.groups()]
            values = [x.strip(" '\"") for x in value.split(symbols.COMMA)]
            self.operators.append([name] + values)

    def get(self, key, default=None):
        """
        Get the value of a specific key.

        :param key str: the key to be searched
        :param default str: the default value if the key is not found
        :return str: the value of the key
        """
        for name, value in self.operators:
            if name == key:
                return value
        return default


class Tag(CheckParser):
    """
    The class to parse the tag file.
    """

    NAME = 'tag'
    SLOW = 'slow'
    TIME_FORMAT = '%H:%M:%S'
    TIME_ZERO = datetime.datetime.strptime('00:00:00', TIME_FORMAT)

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        self.options = options

    def isSlow(self):
        """
        Whether the test is slow.

        :param threshold float: the threshold in seconds to be considered as slow
        :return bool: Whether or not the test is slow.
        """
        value = self.get(self.SLOW)
        if value is None:
            return False
        hms = datetime.datetime.strptime(value, self.TIME_FORMAT)
        delta = hms - self.TIME_ZERO
        return delta.total_seconds() > self.options.slow


class Job(task.Job):
    """
    The class to setup a job cmd for the integration test.
    """

    NAME = 'cmd'
    JOBNAME_RE = re.compile('.* +(.*)_(driver|workflow).py( +.*)?$')
    SEP = symbols.SEMICOLON
    PRE_RUN = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fparser = None
        self.comment = None

    def setArgs(self):
        """
        Set arguments from the input file.
        """
        self.fparser = CmdParser(self.job.statepoint[FLAG_DIR])
        self.fparser.setArgs()
        self.args = self.fparser.args

    def removeUnkArgs(self):
        """
        Remove unknown arguments as the comment.
        """
        self.fparser.removeUnkArgs()
        self.args = self.fparser.args
        self.comment = self.fparser.comment

    def setName(self):
        """
        Set the jobname of the known args.
        """
        for idx, cmd in enumerate(self.args):
            if cmd.startswith('#'):
                continue
            if self.FLAG_JOBNAME in cmd:
                continue
            match = self.JOBNAME_RE.match(cmd)
            if not match:
                continue
            cmd += f" {self.FLAG_JOBNAME} {match.groups()[0]}"
            self.args[idx] = cmd

    def addQuote(self):
        """
        Add quotes for str with special characters.
        """
        for idx, cmd in enumerate(self.args):
            cmd = [self.quoteArg(x.strip()) for x in cmd.split()]
            self.args[idx] = symbols.SPACE.join(cmd)

    def getCmd(self, write=True):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :return str: the command as str
        """
        name = os.path.basename(self.job.statepoint[FLAG_DIR])
        msg = f"{name}: {self.comment}" if self.comment else name
        pre_cmd = [f"echo \'# {msg}\'"]
        return super().getCmd(write=write, pre_cmd=pre_cmd, name=name)

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return bool(self.doc.get(self.OUTFILE))


class Cmd(task.BaseTask):

    JobClass = Job


class EXIST:
    """
    The class to perform file existence check.
    """

    def __init__(self, *args, job=None):
        """
        :param original str: the original filename
        :param args str: the target filenames
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.targets = [x.strip().strip('\'"') for x in args]
        self.job = job

    def run(self):
        """
        The main method to check the existence of files.
        """
        for target in self.targets:
            if not os.path.isfile(self.job.fn(target)):
                raise FileNotFoundError(f"{self.job.fn(target)} not found")


class NOT_EXIST(EXIST):
    """
    The class to perform file non-existence check.
    """

    def run(self):
        """
        The main method to check the existence of a file.
        """
        for target in self.targets:
            if os.path.isfile(self.job.fn(target)):
                raise FileNotFoundError(f"{self.job.fn(target)} found")


class CMP:
    """
    The class to perform file comparison.
    """

    def __init__(self, original, target, job=None):
        """
        :param original str: the original filename
        :param target str: the target filename
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.orignal = original.strip().strip('\'"')
        self.target = target.strip().strip('\'"')
        self.job = job

    def run(self):
        """
        The main method to compare files.
        """
        self.orignal = os.path.join(self.job.statepoint[FLAG_DIR],
                                    self.orignal)
        if not os.path.isfile(self.orignal):
            raise FileNotFoundError(f"{self.orignal} not found")
        self.target = self.job.fn(self.target)
        if not os.path.isfile(self.target):
            raise FileNotFoundError(f"{self.target} not found")
        if not filecmp.cmp(self.orignal, self.target):
            raise ValueError(f"{self.orignal} and {self.target} are different")


class CheckJob(task.BaseJob):
    """
    The class to check the results for one cmd integration test.
    """

    CMD = {'cmp': CMP, 'exist': EXIST, 'not_exist': NOT_EXIST}
    MSG = 'msg'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = []

    def run(self):
        """
        Main method to get the results.
        """
        self.setOperators()
        self.exeOperators()

    def setOperators(self):
        """
        Set the operaters from the check file.
        """
        fparser = CheckParser(self.job.statepoint[FLAG_DIR])
        fparser.run()
        self.operators = fparser.operators

    def exeOperators(self):
        """
        Execute all operators. Raise errors during operation if one failed.
        """
        self.doc[self.MSG] = False
        ops = [symbols.SPACE.join(x) for x in self.operators]
        name = os.path.basename(self.job.statepoint[FLAG_DIR])
        print(f"# {name}: Checking {symbols.COMMA.join(ops)}")
        for operator in self.operators:
            try:
                self.execute(operator)
            except (FileNotFoundError, KeyError, ValueError) as err:
                self.doc[self.MSG] = str(err)

    def execute(self, operator):
        """
        Lookup the command class and execute.

        :param operator list of str: the operator to be executed.
            For example, [name, arg1, arg2, ..]
        """
        name = operator[0]
        try:
            runner_class = self.CMD[name]
        except KeyError:
            raise KeyError(
                f'{name} is one unknown command. Please select from '
                f'{self.CMD.keys()}')
        runner = runner_class(*operator[1:], job=self.job)
        runner.run()

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.MSG in self.doc


class Result(task.BaseTask):
    """
    Class to parse the check file and execute the inside operations.
    """

    JobClass = CheckJob
