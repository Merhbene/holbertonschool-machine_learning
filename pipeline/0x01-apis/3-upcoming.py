#!/usr/bin/env python3
"""
    Barely an API project.
"""
import requests
import time


if __name__ == "__main__":
    """ prints location of user specified as cli arg """
    url = 'https://api.spacexdata.com/v4/launches'
    r = requests.get(url)
    launches_list = r.json()
    # print(len())
    rocket_launches = {}
    for launch in launches_list:
        rocket_id = launch.get('rocket')
        rocket_url = 'https://api.spacexdata.com/v4/rockets/' + rocket_id
        r_rocket = requests.get(rocket_url)
        j_rocket = r_rocket.json()
        rocket_name = j_rocket.get('name')
        if rocket_name not in rocket_launches:
            rocket_launches.update({rocket_name: 1})
        else:
            rocket_launches[rocket_name] += 1
    for k, v in reversed(sorted(rocket_launches.items(), key=lambda k: k[1])):
        print("{}: {}".format(k, v))