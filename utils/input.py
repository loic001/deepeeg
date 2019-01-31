import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def list_selection_input(arr):
    assert arr
    indexed = [(index, item) for index, item  in enumerate(arr)]
    while True:
        for ele1,ele2 in indexed:
            sys.stdout.write("{:<5}{:<11}\n".format(ele1,ele2))
        choice = int(input())
        if choice in [i[0] for i in indexed]:
            return [i[1] for i in indexed if i[0] == choice][0]
        else:
            sys.stdout.write("Please choose a valid index\n\n")
