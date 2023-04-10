import mendeleev
from sqlalchemy.orm.session import close_all_sessions


def element(*args, **kwargs):
    ele = mendeleev.element(*args, **kwargs)
    # mendeleev established sqlite3 connections, which throw out ProgrammingError
    # SQLite objects created in a thread can only be used in that same thread.
    close_all_sessions()
    return ele
