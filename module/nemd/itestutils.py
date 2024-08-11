import os
import re
import filecmp
from nemd import task
from nemd import symbols

FLAG_DIR = '-dir'


class Job(task.Job):
    """
    The class to setup a job cmd for the integration test.
    """

    JOBNAME_RE = re.compile('.* +(.*)_(driver|workflow).py( +.*)?$')
    POUND = symbols.POUND
    CMD = 'cmd'
    PRE_RUN = None
    SEP = symbols.SEMICOLON

    def __init__(self, *args, **kwargs):
        """
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        """
        super().__init__(*args, **kwargs)
        # each cmd job has a unique name based on the directory name
        self.name = os.path.basename(self.job.statepoint[FLAG_DIR])
        self.comment = None

    def setArgs(self):
        """
        Set arguments.
        """
        with open(os.path.join(self.job.statepoint[FLAG_DIR], self.CMD)) as fh:
            self.args = [x.strip() for x in fh.readlines() if x.strip()]

    def removeUnkArgs(self):
        """
        Remove unknown arguments.
        """
        comments = [x for x in self.args if x.startswith(self.POUND)]
        comment = [x.strip(self.POUND).strip() for x in comments]
        self.comment = symbols.SPACE.join(comment)
        self.args = [x for x in self.args if not x.startswith(self.POUND)]

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
            jobname = match.groups()[0]
            cmd += f" {self.FLAG_JOBNAME} {jobname}"
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
        msg = f"{self.name}: {self.comment}" if self.comment else self.name
        pre_cmd = [f"echo \'# {msg}\'"]
        return super().getCmd(write=write, pre_cmd=pre_cmd)

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


class ResultJob(task.BaseJob):
    """
    The class to check the results for one cmd integration test.
    """

    CMD_BRACKET_RE = '\s.*?\(.*?\)'
    PAIRED_BRACKET_RE = '\(.*?\)'
    CMD = {'cmp': CMP, 'exist': EXIST, 'not_exist': NOT_EXIST}
    MSG = 'msg'
    CHECK = 'check'
    AND_RE = r'and\s+'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line = None
        self.operators = []

    def run(self):
        """
        Main method to get the results.
        """
        self.setLine()
        self.parserLine()
        self.executeOperators()

    def setLine(self):
        """
        Set the one line command by locating, reading, and cleaning the check file.
        """
        with open(os.path.join(self.job.statepoint[FLAG_DIR],
                               self.CHECK)) as fh:
            lines = [x.strip() for x in fh.readlines()]
        operators = [x for x in lines if not x.startswith(symbols.POUND)]
        self.line = ' ' + ' '.join(operators)

    def parserLine(self):
        """
        Parse the one line command to get the operators.
        """
        for operator in re.finditer(self.CMD_BRACKET_RE, self.line):
            operator = operator.group().strip()
            operator = re.sub(self.AND_RE, '', operator)
            self.operators.append(operator)

    def executeOperators(self):
        """
        Execute all operators. Raise errors during operation if one failed.
        """
        self.doc[self.MSG] = False
        print(
            f"{self.job.statepoint[FLAG_DIR]}: {symbols.COMMA.join(self.operators)}"
        )
        for operator in self.operators:
            try:
                self.execute(operator)
            except (FileNotFoundError, KeyError, ValueError) as err:
                self.doc[self.MSG] = str(err)

    def execute(self, operator):
        """
        Lookup the command class and execute.
        """
        bracketed = re.findall(self.PAIRED_BRACKET_RE, operator)[0]
        cmd = operator.replace(bracketed, '')
        try:
            runner_class = self.CMD[cmd]
        except KeyError:
            raise KeyError(f'{cmd} is one unknown command. Please select from '
                           f'{self.CMD.keys()}')
        runner = runner_class(*bracketed[1:-1].split(','), job=self.job)
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

    JobClass = ResultJob
