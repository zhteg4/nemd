from nemd import nsignac
from flow.project import FlowProject

class FlowProject(FlowProject, nsignac.Project):
    def __init__(self, *args, **kwargs):
        print('FlowProject nproject is created')
        super().__init__(*args, **kwargs)