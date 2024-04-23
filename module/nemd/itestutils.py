import os
import re
import glob
import filecmp
from nemd import task
from nemd import symbols
from nemd import logutils

DIR = 'dir'
CMD = 'cmd'
CHECK = 'check'
AND = 'and'
SUCCESS = task.SUCCESS
MSG = task.MSG


class Integration_Driver(task.BaseTask):

    @staticmethod
    def operator(*args, **kwargs):
        """
        Get the integration test cmd command.

        :return str: the command to run a task.
        """
        integration_driver = Integration_Driver(*args, **kwargs)
        return integration_driver.getCmd()

    def getCmd(self, write=True):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :return str: the command as str
        """
        cmd_file = os.path.join(self.job.document[DIR], CMD)
        with open(cmd_file) as fh:
            lines = [x.strip() for x in fh.readlines()]
        comment = symbols.COMMA.join([x for x in lines if x.startswith('#')])
        cmd = symbols.SEMICOLON.join(
            [x for x in lines if not x.startswith('#')])
        if write:
            with open(f"{self.job.statepoint[self.STATE_ID]}_cmd", 'w') as fh:
                fh.write(cmd)
        return f"echo \"{os.path.basename(self.job.document[DIR])} {comment}\"; {cmd}"

    @classmethod
    def post(cls, job, name=None):
        """
        The function to determine whether the main command has been executed.

        :param job 'signac.contrib.job.Job': the job object
        :return bool: whether the main command has been executed.

        NOTE：This should be modified when using with slurm schedular.
        """
        filenames = glob.glob(
            job.fn(f"{symbols.WILD_CARD}{logutils.DRIVER_LOG}"))
        if not filenames:
            return False
        jobnames = [
            os.path.basename(x).replace(logutils.DRIVER_LOG, '')
            for x in filenames
        ]
        success = [cls.success(job, x) for x in jobnames]
        return all(success)


class EXIST:
    """
    The class to perform file existence check.
    """

    def __init__(self, target, job=None):
        """
        :param original str: the original filename
        :param target str: the target filename
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.target = target.strip().strip('\'"')
        self.job = job

    def run(self):
        """
        The main method to check the existence of a file.
        """
        self.target = self.job.fn(self.target)
        if not os.path.isfile(self.target):
            raise FileNotFoundError(f"{self.target} not found")


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


class Results(task.BaseTask):
    """
    Class to parse the check file and execute the inside operations.
    """

    CMD_BRACKET_RE = '\s.*?\(.*?\)'
    PAIRED_BRACKET_RE = '\(.*?\)'
    CMD = {'cmp': CMP, 'exist': EXIST}

    def __init__(self, args, **kwargs):
        """
        :param job 'signac.contrib.job.Job': the signac job
        """
        super().__init__(args, **kwargs)
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
            operator = operator.strip(AND + ' ').strip()
            self.operators.append(operator)

    def executeOperators(self):
        """
        Execute all operators. Raise errors during operation if one failed.
        """
        print(f"{self.job.statepoint[self.STATE_ID]}: "
              f"Analyzing {symbols.COMMA.join(self.operators)}")
        for operator in self.operators:
            self.execute(operator)

    def execute(self, operator):
        """
        Lookup the command class and execute.
        """
        bracketed = re.findall(self.PAIRED_BRACKET_RE, operator)[0]
        cmd = operator.replace(bracketed, '')
        try:
            runner_class = self.CMD[cmd]
        except KeyError:
            raise KeyError(
                f'{cmd} is one unknown command. Please select from {self.CMD.keys()}'
            )
        runner = runner_class(*bracketed[1:-1].split(','), job=self.job)
        runner.run()

    @staticmethod
    def operator(job, *args, **kwargs):
        """
        Get results.
        """
        # from remote_pdb import set_trace; set_trace()
        # nc -tC 127.0.0.1 62500
        results = Results(job)
        try:
            results.run()
        except (FileNotFoundError, KeyError, ValueError) as err:
            job.document[SUCCESS] = False
            job.document[MSG] = str(err)
        else:
            job.document[SUCCESS] = True

    @classmethod
    def post(cls, job, **kwargs):
        """
        The method to question whether the checking process has been performed.

        :param job 'signac.contrib.job.Job': the job object
        :return str: the shell command to execute
        """
        return SUCCESS in job.document
