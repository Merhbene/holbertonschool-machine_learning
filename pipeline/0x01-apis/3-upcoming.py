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

op_and(op_and(copy_files(["2006/18119/main_0.py"]), is_equal("", exec_bash("chmod +x 4-rocket_frequency.py"))), is_equal(dry_exec_bash("timeout 180s ./main_0.py"), dry_exec_bash("timeout 180s ./4-rocket_frequency.py")))

op_and(copy_files(["2006/18119/main_0.py"]), is_equal(exec_bash("timeout 180s ./main_0.py"), exec_bash("timeout 180s ./4-rocket_frequency.py")))

op_and(op_and(copy_files(["2006/18119/main_0.py"]), is_equal("", exec_bash("chmod +x 4-rocket_frequency.py"))), is_equal(exec_bash("timeout 180s ./main_0.py"), exec_bash("timeout 180s ./4-rocket_frequency.py")))

