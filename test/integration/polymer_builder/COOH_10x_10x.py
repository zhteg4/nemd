import json
import os

from flow import FlowProject


def volume_computed(job):
    return job.isfile("volume.txt")

# @FlowProject.label
@FlowProject.post(volume_computed)
@FlowProject.operation
def compute_volume(job):
    volume = job.sp.N * job.sp.kT / job.sp.p
    with open(job.fn("volume.txt"), "w") as file:
        file.write(str(volume) + "\n")

# @FlowProject.pre(volume_computed)
# @FlowProject.post.isfile("data.json")
# @FlowProject.operation
# def store_volume_in_json_file(job):
#     with open(job.fn("volume.txt")) as textfile:
#         data = {"volume": float(textfile.read())}
#         with open(job.fn("data.json"), "w") as jsonfile:
#             json.dump(data, jsonfile)


if __name__ == "__main__":
    # init.py
    import signac

    project = signac.init_project(root=os.getcwd(), workspace='workspace')

    for p in range(1, 3):
        sp = {"p": p, "kT": 1.0, "N": 1000}
        job = project.open_job(sp)
        job.init()
    FlowProject().main()