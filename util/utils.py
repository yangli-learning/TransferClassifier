def str2bool(s):
    if s == 'False' or s == 'false':
        return False
    elif s == 'True' or s == 'true':
        return True
    else:
        raise Exception('s should be string: True, true or false, False')
