# -*- coding: utf-8 -*-

def generate_edge_list(data):
    edge_list = []
    for user in data.keys():
        for trustee in data[user]:
            edge_list.append((user, trustee))

    return edge_list


def get_user_item(user_list, data):
    '''
    get itmes of user
    :param user_list:
    :param data:
    :return:
    '''
    user_item_list = []
    for user in user_list:
        u_items = data[user]
        temp = [user] + u_items
        user_item_list.append(temp)
    return user_item_list



