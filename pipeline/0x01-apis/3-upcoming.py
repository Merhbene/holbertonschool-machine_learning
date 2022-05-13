#!/usr/bin/env python3
'''3. What will be next?'''
import requests


def get_launches():
    '''return launches'''
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    request = requests.get(url).json()
    request.sort(key=lambda json: json['date_unix'])
    latest = request[0]
    launch_name = latest['name']
    launch_date = latest['date_local']
    rocket_name = requests.get(
        'https://api.spacexdata.com/v4/rockets/' + latest['rocket']
    ).json()['name']
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' + latest['launchpad']
    ).json()
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']
    print('{} ({}) {} - {} ({})'.format(
        launch_name,
        launch_date,
        rocket_name,
        launchpad_name,
        launchpad_locality
    ))


if __name__ == '__main__':
    get_launches()
