def read_file(filename):
    with (open(filename)) as f:
        lines=f.read
        return lines

def initalize_file(filepath):
    with open(filepath, "w") as f:
        f.write("")
    f.close()


def append_to_file(row,filepath):
    with open(filepath,'a') as f:
        f.write(row)


