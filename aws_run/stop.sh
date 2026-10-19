#!/bin/bash
# Stop (default; EBS kept, ~US$0.02/h for 150 GB gp3) or --terminate (deletes the disk).
cd "$(dirname "$0")"; source common.sh
ID=$(iid); [ -n "$ID" ] || { echo "no v2d-chrome instance"; exit 0; }
if [ "${1:-}" = --terminate ]; then
  $PY -c "import boto3,sys;boto3.client('ec2').terminate_instances(InstanceIds=[sys.argv[1]])" "$ID"; echo "terminating $ID"
else
  $PY -c "import boto3,sys;boto3.client('ec2').stop_instances(InstanceIds=[sys.argv[1]])" "$ID"; echo "stopping $ID"
fi
