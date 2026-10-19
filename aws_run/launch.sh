#!/bin/bash
# PAID: starts one g5.xlarge (~US$1.0/h on-demand in Sydney). Reuses a stopped v2d-chrome instance if one exists.
# Then copies aws_run + sample parquet, runs setup.sh and starts inference (nohup, resumable).
# Usage: bash launch.sh [--dry-run]     (needs AWS_ACCESS_KEY_ID/SECRET and HF_TOKEN in env)
set -euo pipefail
cd "$(dirname "$0")"; source common.sh
DRY=${1:-}
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN missing"; exit 1; }
ID=$(iid)
if [ -z "$ID" ]; then
  ID=$($PY - "$DRY" <<'P'
import boto3, sys, botocore
ec2 = boto3.client("ec2")
sg = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["v2d-chrome-ssh"]}])["SecurityGroups"][0]["GroupId"]
kw = dict(ImageId="ami-0d348852eab53fad3", InstanceType="g5.xlarge", KeyName="v2d-chrome", MinCount=1, MaxCount=1,
          SecurityGroupIds=[sg], UserData=open("/workspace/v2d/watchdog_userdata.sh").read(),
          InstanceInitiatedShutdownBehavior="stop",
          BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True}}],
          TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "v2d-chrome"}]}])
if sys.argv[1] == "--dry-run":
    try: ec2.run_instances(DryRun=True, **kw)
    except botocore.exceptions.ClientError as e: print("DRYRUN:", e.response["Error"]["Code"], file=sys.stderr)
    print("dry"); sys.exit()
print(ec2.run_instances(**kw)["Instances"][0]["InstanceId"])
P
)
  [ "$ID" = dry ] && exit 0
  echo "launched $ID"
else
  [ "$DRY" = --dry-run ] && { echo "would start existing $ID"; exit 0; }
  echo "starting existing $ID"; $PY -c "import boto3,sys;boto3.client('ec2').start_instances(InstanceIds=[sys.argv[1]])" "$ID"
fi
$PY -c "import boto3,sys;boto3.client('ec2').get_waiter('instance_status_ok').wait(InstanceIds=[sys.argv[1]])" "$ID"
IP=$(ipof "$ID"); echo "$ID $IP" > .instance; echo "up: $IP"
# box egress IP changes between sessions: make sure the SG lets this box in on port 22
MYIP=$(curl -s https://checkip.amazonaws.com)
$PY - "$MYIP" <<'P2'
import boto3, sys, botocore
ec2 = boto3.client("ec2"); ip = sys.argv[1] + "/32"
g = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["v2d-chrome-ssh"]}])["SecurityGroups"][0]
if not any(r["CidrIp"] == ip for p in g["IpPermissions"] for r in p["IpRanges"]):
    ec2.authorize_security_group_ingress(GroupId=g["GroupId"], IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": ip, "Description": "v2d box"}]}])
    print("SG: allowed", ip)
P2
for i in $(seq 30); do ssh $SSHO $SSH_USER@$IP true 2>/dev/null && break; sleep 10; done
ssh $SSHO $SSH_USER@$IP "mkdir -p ~/v2d"
scp $SSHO -r ../aws_run $V2D/kitlatest/v2d_submission_kit/data/track_1_sample_submission.parquet $SSH_USER@$IP:~/v2d/
# token goes only into the remote process env (not written to disk on the instance)
ssh $SSHO $SSH_USER@$IP "HF_TOKEN=$HF_TOKEN bash ~/v2d/aws_run/setup.sh" 2>&1 | tail -20
ssh $SSHO $SSH_USER@$IP "bash ~/v2d/aws_run/run_remote.sh"
echo "inference started; watch: bash status.sh ; pull: bash fetch.sh ; stop: bash stop.sh"
