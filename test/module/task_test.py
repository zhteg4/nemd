import pytest
from nemd import task
from nemd import jobutils


class TestBaseJob:

    @pytest.fixture
    def job(self):
        return task.BaseJob(jobutils.Job())

    def testMessage(self, job):
        assert job.message is None
        job.message = False
        assert job.message is False
