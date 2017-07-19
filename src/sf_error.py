# coding=utf-8
# hold any kind of error.
# Zhang Ji, 20160514


class sf_error(RuntimeError):
    _traceback_ = uniqueList()

    def __init__(self, ierr: int = 0, err_mesg: str = ''):
        self.ierr = ierr
        self.err_mesg = err_mesg
        RuntimeError.__init__(self, self.ierr, self.err_mesg)

    def __nonzero__(self):
        return self.ierr != 0

    def __repr__(self):
        return 'StokesFlow.Error: ' + self.err_mesg

    def __str__(self):
        return 'StokesFlow.Error: ' + self.err_mesg
