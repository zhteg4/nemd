import os
import glob
import filecmp
from nemd import symbols
from nemd import environutils
from nemd.nflow import FlowProject

DIR = 'dir'
CHECKED = 'checked'
CMD = 'cmd'


@FlowProject.label
def cmd_completed(job):
    return len(glob.glob(job.fn(symbols.WILD_CARD))) > 1


@FlowProject.post(cmd_completed)
@FlowProject.operation(cmd=True)
def run_cmd(job):
    test_cmd_file = os.path.join(job.document[DIR], CMD)
    with open(test_cmd_file) as fh:
        lines = [x.strip() for x in fh.readlines()]
    cmd = symbols.SEMICOLON.join(lines)
    return f"cd {job.path}; {cmd}; cd -"

def checked(job):
    return CHECKED in job.document

@FlowProject.pre.after(run_cmd)
@FlowProject.post(cmd_completed)
@FlowProject.operation
def check(job):
    assert filecmp.cmp('polymer_builder.data', job.fn('polymer_builder.data'))
    job.document[CHECKED] = True


if __name__ == "__main__":
    base_dir = environutils.get_integration_test_dir()
    base_dir = os.path.join(base_dir, symbols.WILD_CARD)
    test_dirs = [
        x for x in glob.glob(base_dir)
        if os.path.isdir(x) and os.path.basename(x).isdigit()
    ]
    project = FlowProject.init_project(workspace='workspace')
    for test_dir in test_dirs:
        job = project.open_job({'id': os.path.basename(test_dir)})
        job.document[DIR] = test_dir
        job.init()
    project.run()
    project.print_status(detailed=True)
