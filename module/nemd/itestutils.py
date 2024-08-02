import os
import re
import sh
import filecmp
from nemd import task
from nemd import symbols
from nemd import jobutils

DIR = 'dir'
CMD = 'cmd'
CHECK = 'check'
AND_RE = r'and\s+'


class Job(task.Job):
    """
    The class to setup a job cmd for the integration test.
    """

    STATE_ID = jobutils.STATE_ID
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    JOBNAME_RE = re.compile('.* +(.*)_(driver|workflow).py( +.*)?$')

    def __init__(self, *args, pre_run=None, **kwargs):
        """
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param pre_run: append this str before the driver path
        :type pre_run: str

        """
        super().__init__(*args, pre_run=pre_run, **kwargs)
        self.comments = []
        self.name = self.job.statepoint[self.STATE_ID]

    def setArgs(self):
        """
        Set arguments.
        """
        cmd_file = os.path.join(self.doc[DIR], CMD)
        with open(cmd_file) as fh:
            self.args = [x.strip() for x in fh.readlines()]

    def removeUnkArgs(self):
        """
        Remove unknown arguments.
        """
        self.comments = [x.strip('#') for x in self.args if x.startswith('#')]
        self.args = [x for x in self.args if x and not x.startswith('#')]

    def setName(self):
        """
        Set the jobname of the known args.
        """
        for idx, cmd in enumerate(self.args):
            if cmd.startswith('#'):
                continue
            if self.FLAG_JOBNAME in cmd:
                continue
            jobname = self.JOBNAME_RE.match(cmd).groups()[0]
            cmd += f" {self.FLAG_JOBNAME} {jobname}"
            self.args[idx] = cmd

    def addQuote(self):
        """
        Add quotes for str with special characters.
        """
        quote_needed = lambda x: self.SPECIAL_CHAR_RE.search(
            x) and not self.QUOTED_RE.match(x)
        for idx, cmd in enumerate(self.args):
            cmd = [f"'{x}'" if quote_needed(x) else x for x in cmd.split()]
            self.args[idx] = ' '.join(cmd)

    def getCmd(self, write=True):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :return str: the command as str
        """
        comment = symbols.COMMA.join(self.comments)
        pre = [f"echo \'# {os.path.basename(self.doc[DIR])}: {comment}\'"]
        return super().getCmd(write=write, sep=symbols.SEMICOLON, pre_cmd=pre)


class Integration(task.BaseTask):

    JobClass = Job

    @classmethod
    def post(cls, job, name=None):
        """
        The job is considered finished when the post-conditions return True.

        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return: True if the post-conditions are met
        :rtype: bool
        """
        return super().post(job, name=None)


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
        self.orignal = os.path.join(self.job.document[DIR], self.orignal)
        if not os.path.isfile(self.orignal):
            raise FileNotFoundError(f"{self.orignal} not found")
        self.target = self.job.fn(self.target)
        if not os.path.isfile(self.target):
            raise FileNotFoundError(f"{self.target} not found")
        if not filecmp.cmp(self.orignal, self.target):
            raise ValueError(f"{self.orignal} and {self.target} are different")


class ResultJob(task.Job):
    """
    The class to check the results for one cmd integration test.
    """

    CMD_BRACKET_RE = '\s.*?\(.*?\)'
    PAIRED_BRACKET_RE = '\(.*?\)'
    CMD = {'cmp': CMP, 'exist': EXIST, 'not_exist': NOT_EXIST}
    MSG = task.MSG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line = None
        self.operators = []
        self.doc[self.MSG] = False

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
        check_file = os.path.join(self.job.document[DIR], CHECK)
        with open(check_file) as fh:
            lines = [x.strip() for x in fh.readlines()]
        operators = [x for x in lines if not x.startswith(symbols.POUND)]
        self.line = ' ' + ' '.join(operators)

    def parserLine(self):
        """
        Parse the one line command to get the operators.
        """
        for operator in re.finditer(self.CMD_BRACKET_RE, self.line):
            operator = operator.group().strip()
            operator = re.sub(AND_RE, '', operator)
            self.operators.append(operator)

    def executeOperators(self):
        """
        Execute all operators. Raise errors during operation if one failed.
        """
        print(f"{self.job.statepoint[self.STATE_ID]}: "
              f"Analyzing {symbols.COMMA.join(self.operators)}")
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


class Results(task.BaseTask):
    """
    Class to parse the check file and execute the inside operations.
    """

    JobClass = ResultJob

    @classmethod
    def post(cls, job, **kwargs):
        """
        The method to question whether the checking process has been performed.

        :param job 'signac.contrib.job.Job': the job object
        :return str: the shell command to execute
        """
        return ResultJob.MSG in job.document
