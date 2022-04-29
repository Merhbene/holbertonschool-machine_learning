#!/usr/bin/env python3
""" 
By using the SpaceX API (https://github.com/r-spacex/SpaceX-API/blob/master/docs/v4/README.md), 
display the upcomping launch:

- name of the launch
- the date (in local time)
- the rocket name
- the name (with locality) of the launchpad

 """
import requests
import sys
import time


if __name__ == '__main__':
    r_launches = requests.get("https://api.spacexdata.com/v4/launches/upcoming").json()
    first_launch = None
    date_unix = None
    for r_launch in r_launches:
        if date_unix is None or date_unix > r_launch.get('date_unix', 0):
            first_launch = r_launch
            date_unix = r_launch.get('date_unix', 0)

    r_rocket = requests.get("https://api.spacexdata.com/v4/rockets/{}".format(first_launch.get('rocket'))).json()
    r_launchpad = requests.get("https://api.spacexdata.com/v4/launchpads/{}".format(first_launch.get('launchpad'))).json()
    print("{} ({}) {} - {} ({})".format(first_launch.get('name'), first_launch.get('date_local'), r_rocket.get('name'), r_launchpad.get('name'), r_launchpad.get('locality')))
