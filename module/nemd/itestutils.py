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
    CMD_BRACKET_RE = '.*?\(.*?\)'
    NAME_BRACKET_RE = re.compile('(?:(?:^(?:\s+)?)|(?:\s+))(.*?)\\((.*?)\\)')
    AND_NAME_RE = re.compile('^and\s+(.*)')

    def __init__(self, path=None, job=None):
        """
        :param path str: the path containing the file
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.path = path
        self.job = job
        path_dir = self.path if path else self.job.statepoint[FLAG_DIR]
        self.pathname = os.path.join(path_dir, self.NAME)
        self.args = None
        self.comment = None

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
    The class to setup a job cmd for the integration test.
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
        Main method to setup the job.
        """
        self.parse()
        self.setName()
        self.addQuote()

    def parse(self):
        """
        Set arguments from the input file.
        """
        parser = Cmd(job=self.job)
        parser.parse()
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


class Opr(Cmd):

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

    NAME = 'check'
    CMD = {'cmp': CMP, 'exist': EXIST, 'not_exist': NOT_EXIST}

    def run(self):
        """
        Parse the file, set the operators, and execute the operators.
        """
        self.parse()
        self.check()

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
    The class to parse the file and set the operators.
    """

    def run(self):
        """
        Main method to run.
        """
        try:
            Check(job=self.job).run()
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
    """
    Class to parse the check file and execute the inside operations.
    """

    JobClass = CheckJob


class Tag(Opr):
    """
    The class to parse the tag file.
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
        self.parse()
        self.setLogs()
        self.setSlow()
        self.setLabel()
        self.write()

    def selected(self):
        """
        Select the operators by the options.

        :return bool: Whether the test is selected.
        """
        if self.options is None or self.options.slow is None:
            return True
        self.parse()
        return not self.slow

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
        delta = timeutils.str2timedelta(value)
        return delta.total_seconds() > self.options.slow

    def setLogs(self):
        """
        Set the log files for the job.
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
        total_time = datetime.timedelta()
        # The cmd may run multiple jobs, and workflow job may be included.
        for log in self.logs:
            total_time += log.task_time
        job_time = timeutils.delta2str(total_time)
        self.set(self.SLOW, job_time)

    def setLabel(self):
        """
        Set the label of the job.
        """
        label = self.get(self.LABEL, [])
        for log in self.logs:
            label.append(log.options.default_name)
        if label:
            self.set(self.LABEL, *set(label))

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

    def run(self):
        """
        Main method to run.
        """
        Tag(job=self.job).run()
        self.message = False


class TagTask(task.BaseTask):

    JobClass = TagJob
