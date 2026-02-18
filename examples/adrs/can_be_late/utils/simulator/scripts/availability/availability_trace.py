# Adopted from https://github.com/stanford-futuredata/training_on_a_dime/blob/master/scripts/aws/availability.py
import argparse
from datetime import datetime
import signal
import json
import os
import subprocess
import sys
import time

import ray


configs = {}

CLOCK = 0
instance_types = {
    ("v100", 1): "p3.2xlarge",
    ("v100", 4): "p3.8xlarge",
    ("v100", 8): "p3.16xlarge",
    ("k80", 1): "p2.xlarge",
    ("k80", 8): "p2.8xlarge",
    ("k80", 16): "p2.16xlarge",
    ("t4", 1): "g4dn.2xlarge",
    ("t4", 4): "g4dn.12xlarge",
    ("t4", 8): "g4dn.metal",
}

def signal_handler(sig, frame):
    global configs
    # Clean up all instances when program is interrupted.
    for (zone, gpu_type, num_gpus) in configs:
        [instance_id, _] = configs[(zone, gpu_type, num_gpus)]
        if instance_id is not None:
            delete_spot_instance(zone, instance_id)
    sys.exit(0)

def delete_spot_instance(zone, instance_id):
    command = """aws ec2 terminate-instances --instance-ids %(instance_id)s""" % {
        "instance_id": instance_id,
    }
    try:
        output = subprocess.check_output(command, shell=True)
        print("[%s] Successfully deleted instance %s" % (
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'), instance_id))
    except:
        return

@ray.remote(num_cpus=0)
def launch_spot_instance(zone, gpu_type, num_gpus):
    instance_type = instance_types[(gpu_type, num_gpus)]
    with open("specification.json.template", 'r') as f1, open(f"specification-{zone}-{gpu_type}-{num_gpus}.json", 'w') as f2:
        template = f1.read()
        specification_file = template % (instance_type, zone)
        f2.write(specification_file)
    command = f"""aws ec2 request-spot-instances --instance-count 1 --type one-time --launch-specification file://specification-{zone}-{gpu_type}-{num_gpus}.json"""
    instance_id = None
    spot_instance_request_id = None
    try:
        try:
            print("[%s] Trying to create instance with %d GPU(s) of type %s in zone %s" % (
                datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                num_gpus, gpu_type, zone), file=sys.stderr)
            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)
            spot_instance_request_id = return_obj["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
            command = """aws ec2 describe-spot-instance-requests --spot-instance-request-id %s""" % (
                spot_instance_request_id)
            time.sleep(30)
            output = subprocess.check_output(command, shell=True).decode()
            return_obj = json.loads(output)
            instance_id = return_obj["SpotInstanceRequests"][0]["InstanceId"]
            print("[%s] Created instance %s with %d GPU(s) of type %s in zone %s" % (
                datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                instance_id, num_gpus, gpu_type, zone))
            return True
        except Exception as e:
            print(type(e), e)
            pass
        
        print("[%s] Instance with %d GPU(s) of type %s creation in zone %s failed" % (
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'), num_gpus, gpu_type, zone))
        return False
    finally:
        if instance_id is not None:
            delete_spot_instance(zone, instance_id)
        if spot_instance_request_id is not None and instance_id is None:
            command = """aws ec2 cancel-spot-instance-requests --spot-instance-request-ids %s""" % (
                spot_instance_request_id)
            subprocess.check_output(command, shell=True)
            print("[%s] Successfully cancelled spot request %s" % (
                datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'), spot_instance_request_id))
        print('ready')


def main(args):
    global configs
    global CLOCK
    ray.init()
    
    for zone in args.zones:
        for gpu_type in args.gpu_types:
            for num_gpus in args.all_num_gpus:
                configs[(zone, gpu_type, num_gpus)] = [None, False]

    start_date = datetime.now().strftime('%Y-%m-%dT%H-%M')
    folder = f'traces/{start_date}'
    os.makedirs(folder, exist_ok=True)
    while True:
        print(f"Clock: {CLOCK}")
        # Spin in a loop; try to launch spot instances of particular type if
        # not running already. Check on status of instances, and update to
        # "not running" as needed.
        workers = [launch_spot_instance.remote(*config) for config in configs]
        dt = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')

        time.sleep(600)
        ready, not_ready = ray.wait(workers, timeout=0.1, num_returns=len(workers))
        assert len(not_ready) == 0, (ready, not_ready)
        for i, (zone, gpu_type, num_gpus) in enumerate(configs):
            fail = int(not ray.get(ready[i]))
            with open(f'{folder}/{zone}_{gpu_type}_{num_gpus}.txt', 'a') as f:
                print(f'{CLOCK},{dt},{fail}', file=f)
        CLOCK += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Get AWS spot instance availability')
    parser.add_argument('--zones', type=str, nargs='+',
                        default=["us-west-2a", "us-west-2b"],
                        help='AWS availability zones')
    parser.add_argument('--gpu_types', type=str, nargs='+',
                        default=["v100", "k80"],
                        help='GPU types')
    parser.add_argument('--all_num_gpus', type=int, nargs='+',
                        default=[1, 8],
                        help='Number of GPUs per instance')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    main(args)
