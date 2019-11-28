def are_lists_isomorphic(list_1, list_2):
    if len(list_1) != len(list_2):
        return False

    distinct_elements_1 = set(list_1)
    distinct_elements_2 = set(list_2)

    if len(distinct_elements_1) != len(distinct_elements_2):
        return False

    mappings = [(item_1, item_2) for (item_1, item_2)
                in zip(list_1, list_2)]
    distinct_mappings = set(mappings)

    return len(distinct_elements_1) == len(distinct_mappings)
