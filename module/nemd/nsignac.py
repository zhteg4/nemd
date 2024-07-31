from signac import Project
from signac import job

JOB_ID_LENGTH = 32

class Job(job.Job):
    def __init__(self, *args, **kwargs):
        print('Job nsignac is created')
        super().__init__(*args, **kwargs)

    def wa(self, *args, **kwargs):
        print('wa is called')

class Project(Project):
    def __init__(self, *args, **kwargs):
        print('Project nsignac is created')
        super().__init__(*args, **kwargs)

    def open_job(self, statepoint=None, id=None):
        """Get a job handle associated with a state point.

        This method returns the job instance associated with
        the given state point or job id.
        Opening a job by a valid state point never fails.
        Opening a job by id requires a lookup of the state point
        from the job id, which may fail if the job was not
        previously initialized.

        Parameters
        ----------
        statepoint : dict, optional
            The job's unique set of state point parameters (Default value = None).
        id : str, optional
            The job id (Default value = None).

        Returns
        -------
        :class:`~signac.job.Job`
            The job instance.

        Raises
        ------
        KeyError
            If the attempt to open the job by id fails.
        LookupError
            If the attempt to open the job by an abbreviated id returns more
            than one match.

        """
        if (statepoint is None) == (id is None):
            raise ValueError("Either statepoint or id must be provided, but not both.")
        if id is None:
            # Second best case (Job will update self._sp_cache on init)
            return Job(project=self, statepoint=statepoint)
        try:
            # Optimal case (id is in the state point cache)
            return Job(project=self, statepoint=self._sp_cache[id], id_=id)
        except KeyError:
            # Worst case: no state point was provided and the state point cache
            # missed. The Job will register itself in self._sp_cache when the
            # state point is accessed.
            if len(id) < JOB_ID_LENGTH:
                # Resolve partial job ids (first few characters) into a full job id
                job_ids = self._find_job_ids()
                matches = [id_ for id_ in job_ids if id_.startswith(id)]
                if len(matches) == 1:
                    id = matches[0]
                elif len(matches) > 1:
                    raise LookupError(id)
                else:
                    # By elimination, len(matches) == 0
                    raise KeyError(id)
            elif not self._contains_job_id(id):
                # id does not exist in the project data space
                raise KeyError(id)
            return Job(project=self, id_=id)