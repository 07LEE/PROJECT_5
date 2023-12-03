from .arguments import get_train_args

# load name list from name_list.txt
args = get_train_args()
name_list_path = args.name_list_path


# load the name list from file
def get_alias2id(name_list_path):
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    alias2id = {}
    for i, line in enumerate(name_lines):
        for alias in line.strip().split()[1:]:
            alias2id[alias] = i
    return alias2id


def get_id2alias(name_list_path):
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    id2alias = []
    for i, line in enumerate(name_lines):
        id2alias.append(line.strip().split()[1])
    return id2alias
