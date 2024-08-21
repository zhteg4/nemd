import os
import re
import filecmp
import datetime
from nemd import task
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import timeutils

FLAG_DIR = '-dir'


class Cmd:
    """
    The class to parse a cmd test file.
    """

    NAME = 'cmd'
    POUND = symbols.POUND
    CMD_BRACKET_RE = r'.*?\(.*?\)'
    NAME_BRACKET_RE = re.compile(r'(?:(?:^(?:\s+)?)|(?:\s+))(.*?)\\((.*?)\\)')
    AND_NAME_RE = re.compile(r'^and\s+(.*)')

    def __init__(self, dir=None, job=None, delay=False):
        """
        :param dir str: the path containing the file
        :param job 'signac.contrib.job.Job': the signac job instance
        :param delay 'bool': read, parse, and set the operators if False
        """
        self.dir = dir
        self.job = job
        self.delay = delay
        self.args = None
        self.comment = None
        if self.dir is None:
            self.dir = self.job.statepoint[FLAG_DIR]
        self.pathname = os.path.join(self.dir, self.NAME)
        if self.delay:
            return
        self.parse()

    def parse(self):
        """
        Main method to parse one integration test file.
        """
        self.read()
        self.setComment()

    def read(self):
        """
        Set arguments by reading the file.
        """
        if not os.path.isfile(self.pathname):
            return
        with open(self.pathname) as fh:
            self.args = [x.strip() for x in fh.readlines() if x.strip()]

    def setComment(self):
        """
        Take the comment out of the args, and set the comment attribute.
        """
        if self.args is None:
            return
        comments = [x for x in self.args if x.startswith(self.POUND)]
        comment = [x.strip(self.POUND).strip() for x in comments]
        self.comment = symbols.SPACE.join(comment)
        self.args = [x for x in self.args if not x.startswith(self.POUND)]


class CmdJob(task.Job):
    """
    The class to set up a job cmd so that the integration test can run normal
    nemd jobs from the cmd line.
    """

    NAME = 'cmd'
    JOBNAME_RE = re.compile('.* +(.*)_(driver|workflow).py( +.*)?$')
    SEP = symbols.SEMICOLON
    PRE_RUN = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comment = None

    def run(self):
        """
        Main method to set up the job.
        """
        self.parse()
        self.setName()
        self.addQuote()

    def parse(self):
        """
        Set arguments from the input file.
        """
        parser = Cmd(job=self.job)
        self.args = parser.args
        self.comment = parser.comment

    def setName(self):
        """
        Set the cmd job names.
        """
        for idx, cmd in enumerate(self.args):
            match = self.JOBNAME_RE.match(cmd)
            if not match or self.FLAG_JOBNAME in cmd:
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


class CmdTask(task.BaseTask):

    JobClass = CmdJob


class Exist:
    """
    The class to perform file existence check.
    """

    def __init__(self, *args, job=None):
        """
        :param original str: the original filename
        :param args str: the target filenames
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.targets = args
        self.job = job

    def run(self):
        """
        The main method to check the existence of files.
        """
        for target in self.targets:
            if not os.path.isfile(self.job.fn(target)):
                raise FileNotFoundError(f"{self.job.fn(target)} not found")


class Not_Exist(Exist):
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


class Cmp:
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


class Opr(Cmd):
    """
    The class sets the operators in addition to the parsing a file.
    """

    NAME = 'opr'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = []

    def parse(self):
        """
        Parse the file, set the operators, and execute the operators.
        """
        super().parse()
        self.setOperators()

    def setOperators(self):
        """
        Parse the one line command to get the operators.
        """
        if self.args is None:
            return
        for match in re.finditer(self.NAME_BRACKET_RE, ' '.join(self.args)):
            name, value = [x.strip("'\"") for x in match.groups()]
            match = self.AND_NAME_RE.match(name)
            if match:
                name = match.groups()[0]
            values = [x.strip(" '\"") for x in value.split(symbols.COMMA)]
            self.operators.append([name] + [x for x in values if x])


class Check(Opr):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    NAME = 'check'
    CMD = {'cmp': Cmp, 'exist': Exist, 'not_exist': Not_Exist}

    def check(self):
        """
        Check the results by execute all operators. Raise errors if any failed.
        """
        ops = [symbols.SPACE.join(x) for x in self.operators]
        name = os.path.basename(os.path.dirname(self.pathname))
        print(f"# {name}: Checking {symbols.COMMA.join(ops)}")
        for operator in self.operators:
            self.execute(operator)

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


class CheckJob(task.BaseJob):
    """
    The job class to parse the check file, run the operators, and set the job
    message.
    """

    def run(self):
        """
        Main method to run.
        """
        try:
            Check(job=self.job).check()
        except (FileNotFoundError, KeyError, ValueError) as err:
            self.message = str(err)
        else:
            self.message = False

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.name in self.doc[self.MESSAGE]


class CheckTask(task.BaseTask):

    JobClass = CheckJob


class Tag(Opr):
    """
    The class parses and interprets the tag file. The class also generates new
    tag file (or updates the existing one).
    """

    NAME = 'tag'
    SLOW = 'slow'
    LABEL = 'label'

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, **kwargs)
        self.options = options
        self.logs = []

    def run(self):
        """
        Main method to run.
        """
        self.setLogs()
        self.setSlow()
        self.setLabel()
        self.write()

    def selected(self):
        """
        Select the operators by the options.

        :return bool: Whether the test is selected.
        """
        not_slow = self.options.slow is None or not self.slow
        has_label = self.options.label is None or set(
            self.options.label).intersection(self.get(self.LABEL, []))
        return all([not_slow, has_label])

    @property
    def slow(self):
        """
        Whether the test is slow.

        :return bool: Whether the test is slow.
        """
        if self.options is None or self.options.slow is None:
            return False
        value = self.get(self.SLOW)
        if value is None:
            return False
        delta = timeutils.str2delta(value[0])
        return delta.total_seconds() > self.options.slow

    def setLogs(self):
        """
        Set the log readers.
        """
        logfiles = self.job.doc.get(jobutils.LOGFILE)
        if logfiles is None:
            return
        for logfile in logfiles.values():
            self.logs.append(logutils.LogReader(self.job.fn(logfile)))

    def setSlow(self):
        """
        Set the slow tag with the total job time from the driver log files.
        """
        total = sum([x.task_time for x in self.logs], datetime.timedelta())
        job_time = timeutils.delta2str(total)
        self.set(self.SLOW, job_time)

    def setLabel(self):
        """
        Set the label of the job.
        """
        labels = self.get(self.LABEL, [])
        labels += [x.options.default_name for x in self.logs]
        if not labels:
            return
        self.set(self.LABEL, *set(labels))

    def get(self, key, default=None):
        """
        Get the value of a specific key.

        :param key str: the key to be searched
        :param default str: the default value if the key is not found
        :return tuple of str: the value(s)
        """
        for name, *value in self.operators:
            if name == key:
                return value
        return tuple() if default is None else default

    def set(self, key, *value):
        """
        Set the value (and the key) of one operator.

        :param key str: the key to be set
        :param value tuple of str: the value(s) to be set
        """
        key_values = [key, *value]
        for idx, (name, *_) in enumerate(self.operators):
            if name == key:
                self.operators[idx] = key_values
                return
        self.operators.append(key_values)

    def write(self):
        """
        Write the tag file.
        """
        ops = [f"{x[0]}({symbols.COMMA.join(x[1:])})" for x in self.operators]
        name = os.path.basename(os.path.dirname(self.pathname))
        print(f"# {name}: Tagging {symbols.COMMA.join(ops)}")
        with open(self.pathname, 'w') as fh:
            for key, *value in self.operators:
                values = symbols.COMMA.join(value)
                fh.write(f"{key}({values})\n")


class TagJob(CheckJob):
    """
    This job class generates a new tag file (or updates the existing one).
    """

    def run(self):
        """
        Main method to run.
        """
        Tag(job=self.job).run()
        self.message = False


class TagTask(task.BaseTask):

    JobClass = TagJob
