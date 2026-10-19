# shared settings (sourced)
export AWS_DEFAULT_REGION=ap-southeast-2
V2D=/workspace/v2d
PY=$V2D/awsvenv/bin/python
KEY=$V2D/v2d-chrome-syd.pem
AMI=ami-0d348852eab53fad3
ITYPE=${ITYPE:-g5.xlarge}
SG_NAME=v2d-chrome-ssh
KEY_NAME=v2d-chrome
TAG=v2d-chrome
SSH_USER=${SSH_USER:-ubuntu}
SSHO="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o LogLevel=ERROR"
iid() { $PY - <<'P'
import boto3
r=boto3.client("ec2").describe_instances(Filters=[{"Name":"tag:Name","Values":["v2d-chrome"]},
  {"Name":"instance-state-name","Values":["pending","running","stopping","stopped"]}])
ids=[i["InstanceId"] for x in r["Reservations"] for i in x["Instances"]]; print(ids[0] if ids else "")
P
}
ipof() { $PY -c "import boto3,sys;i=boto3.client('ec2').describe_instances(InstanceIds=[sys.argv[1]])['Reservations'][0]['Instances'][0];print(i.get('PublicIpAddress',''))" "$1"; }
