import os
import signac
import filecmp
import subprocess
from nemd import fileutils
from nemd.nflow import FlowProject

CMD = 'run_nemd polymer_builder_driver.py *CC* -cru_num 10 -mol_num 10 -seed 5678'

@FlowProject.operation
def build_polymer(job):
    with fileutils.chdir(job.fn('')):
        subprocess.run(CMD.split())

@FlowProject.pre.after(build_polymer)
@FlowProject.operation
def cmp_data(job):
    assert filecmp.cmp( 'polymer_builder.data', job.fn('polymer_builder.data'))


if __name__ == "__main__":

    project = signac.init_project(root=os.getcwd(), workspace='workspace')
    job = project.open_job({'V':0})
    job.init()
    FlowProject().run()