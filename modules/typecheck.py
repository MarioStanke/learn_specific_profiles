""" Custom function(s) for type checking, to avoid type errors to some extend. """

import logging

def typecheck(obj, expected: str, die: bool = False, log_warnings: bool = True) -> bool:
    """ Check the expected type of an object via a `classname` string member of the object's class. 
    The `classname` should be a class attribute and not part of the `__init__` function to avoid accidental overwriting.

    Arguments:
        obj: object to be checked
        expected (str): expected type
        die (bool): if True, raise an AssertionError if the check fails
        log_warnings (bool): if True (default), log error messages

    Returns:
        bool: True if the check passes, False otherwise

    Raises:
        AssertionError: if the check fails and `die` is True
    """

    if hasattr(obj, 'classname'):
        if obj.classname == expected:
            return True
        else:
            msg = f"[ERROR] >>> Expected obj of type {expected}, got {obj.classname} of type {type(obj)}"
            if die:
                raise AssertionError(msg)
            else:
                if log_warnings:
                    logging.error(msg)
                return False
            
    else:
        msg = f"[ERROR] >>> Expected obj of type {expected}, got {type(obj)}"
        if die:
            raise AssertionError(msg)
        else:
            if log_warnings:
                logging.error(msg)
            return False
        

def typecheck_list(obj, expected: list[str], die: bool = False, log_warnings: bool = True) -> bool:
    """ Check the expected possibilities of types of an object via a `classname` string member of the object's class. 
    See `typecheck` for more details. """
    for ex in expected:
        if not typecheck(obj, ex, die, log_warnings):
            return False
        
    return True


def typecheck_objdict(obj: dict, expected: str, die: bool = False, log_warnings: bool = True) -> bool:
    """ Check the expected types of a dict-representation of an object via a `classname` key, coming from the 
    `classname` class member. See `typecheck` for more details. """
    if 'classname' in obj:
        if obj['classname'] == expected:
            return True
        else:
            msg = f"[ERROR] >>> Expected obj of type {expected}, got {obj['classname']} in obj dict"
            if die:
                raise AssertionError(msg)
            else:
                if log_warnings:
                    logging.error(msg)
                return False
    else:
        msg = "[ERROR] >>> Expected obj of type {expected}, got dict without 'classname' key"
        if die:
            raise AssertionError(msg)
        else:
            if log_warnings:
                logging.error(msg)
            return False


def typecheck_objdict_list(obj: dict, expected: list[str], die: bool = False, log_warnings: bool = True) -> bool:
    """ Check the expected possibilities of types of an object via a `classname` key of the object's dict 
    representation. See `typecheck_objdict` for more details. """
    for ex in expected:
        if not typecheck_objdict(obj, ex, die, log_warnings):
            return False
        
    return True