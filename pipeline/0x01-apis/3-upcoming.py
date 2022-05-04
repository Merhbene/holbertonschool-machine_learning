#!/usr/bin/env python3
"""API module"""
import requests
from datetime import datetime


if __name__ == '__main__':
    base_url = "https://api.spacexdata.com/v4"

    response = requests.get(base_url + "/launches/upcoming")
    content = response.json()
    upcoming_launch = tuple()

    for launch in content:
        launch_name = launch['name']
        launch_date = launch['date_unix']
        rocket_id = launch['rocket']
        lauchpad_id = launch['launchpad']
        if len(upcoming_launch) == 0 or launch_date < upcoming_launch[0]:
            upcoming_launch = (
                launch_date,
                launch_name,
                rocket_id,
                lauchpad_id,
            )

    launch_name = upcoming_launch[1]
    launch_date = datetime.fromtimestamp(upcoming_launch[0]).isoformat()

    response = requests.get(base_url + "/rockets/" + upcoming_launch[2])
    rocket_name = response.json()['name']

    response = requests.get(base_url + "/launchpads/" + upcoming_launch[3])
    launchpad_name = response.json()['name']
    launchpad_loc = response.json()['locality']

    print(
        "{} ({}) {} - {} ({})".format(
            launch_name,
            launch_date,
            rocket_name,
            launchpad_name,
            launchpad_loc
        )
    )