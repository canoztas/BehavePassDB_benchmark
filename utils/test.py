
def create_folder_name(combo_list):
    if len(combo_list) == 0:
        tmp_folder_name = combo_list[0]
    else:
        tmp_folder_name = '-'.join(combo_list)
    return tmp_folder_name