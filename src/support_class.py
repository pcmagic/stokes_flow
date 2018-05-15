import sys
from collections import UserList
import numpy as np

__all__ = ['uniqueList', 'typeList', 'intList', 'floatList',
           'abs_comp', 'abs_construct_matrix',
           'check_file_extension',
           'tube_flatten', ]


class uniqueList(UserList):
    def __init__(self, acceptType=None):
        self._acceptType = acceptType
        super().__init__()

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

        err_msg = 'item ' + repr(other) + ' add muilt times. '
        assert self.count(other) == 0, err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)


class typeList(UserList):
    def __init__(self, acceptType):
        self._acceptType = acceptType
        super().__init__()

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)


class intList(typeList):
    def __init__(self):
        super().__init__(int)


class floatList(typeList):
    def __init__(self):
        super().__init__(float)


class abs_comp:
    def __init__(self, **kwargs):
        need_args = ['name']
        opt_args = {'childType': abs_comp}
        self._kwargs = kwargs
        self.check_args(need_args, opt_args)

        self._father = None
        self._name = kwargs['name']
        self._child_list = uniqueList(acceptType=kwargs['childType'])
        self._create_finished = False
        self._index = -1

    def checkmyself(self):
        for child in self._child_list:
            self._create_finished = child.checkmyself and \
                                    self._create_finished
        return self._create_finished

    def myself_info(self):
        if self._create_finished:
            str = ('%s: %d, %s, create sucessed' % (self.__class__.__name__, self._index, self._name))
        else:
            str = ('%s: %d, %s, create not finished' % (self.__class__.__name__, self._index, self._name))
        return str

    def printmyself(self):
        PETSc.Sys.Print(self.myself_info())
        return True

    def savemyself(self, file_name):
        fid = open(file_name, 'w')
        fid.write(self.myself_info())
        return True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, father):
        self._father = father

    def save_context(self):
        pass

    def restore_context(self):
        pass

    def check_child_index(self, index):
        err_msg = 'wrong index %d, last index is %d. ' % (index, len(self._child_list))
        assert index <= len(self._child_list), err_msg

        err_msg = 'wrong index %d, first index is 0' % (index)
        assert index > 0, err_msg
        return True

    def get_child(self, index):
        self.check_child_index(index)
        return self._child_list[index]

    @property
    def child_list(self):
        return self._child_list

    def __repr__(self):
        return self.myself_info()

    def check_need_args(self, need_args=list()):
        kwargs = self._kwargs
        for key in need_args:
            err_msg = "information about '%s' is necessary for %s.%s. " \
                      % (key, self.__class__.__name__, sys._getframe(2).f_code.co_name)
            assert key in kwargs, err_msg
        return True

    def check_opt_args(self, opt_args=dict()):
        kwargs = self._kwargs
        for key, value in opt_args.items():
            if not key in kwargs:
                kwargs[key] = opt_args[key]
        self._kwargs = kwargs
        return kwargs

    def check_args(self, need_args=list(), opt_args=dict()):
        self.check_need_args(need_args)
        kwargs = self.check_opt_args(opt_args)
        self._kwargs = kwargs
        return kwargs


# abstract class for matrix construction.
class abs_construct_matrix(abs_comp):
    def __init__(self):
        super().__init__(childType=abs_comp)


def check_file_extension(filename, extension):
    if filename[-len(extension):] != extension:
        filename = filename + extension
    return filename


def tube_flatten(container):
    for i in container:
        if isinstance(i, (uniqueList, list, tuple)):
            for j in tube_flatten(i):
                yield j
        else:
            yield i
