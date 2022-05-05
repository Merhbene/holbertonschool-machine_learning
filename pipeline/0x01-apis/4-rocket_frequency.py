#!/usr/bin/env python3
'''4. How many by rocket?'''
import requests


if __name__ == '__main__':
    rockets = {}
    launches_req = requests.get(
        'https://api.spacexdata.com/v4/launches'
    ).json()
    for explorer in launches_req:
        rocket_id = explorer['rocket']
        if rocket_id in rockets:
            rockets[rocket_id] += 1
        else:
            rockets[rocket_id] = 1
    rockets = sorted(
        rockets.items(), key=lambda rocket: rocket[1], reverse=True
    )
    for rocket in rockets:
        rocket_name = requests.get(
            'https://api.spacexdata.com/v4/rockets/' + rocket[0]
        ).json()['name']
        print('{}: {}'.format(
            rocket_name,
            rocket[1]
        ))
