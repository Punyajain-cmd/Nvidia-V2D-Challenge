#!/bin/bash
# Pull finished episode npz files (+ log) back to the box. Safe to run repeatedly.
cd "$(dirname "$0")"; source common.sh; read ID IP < .instance
mkdir -p $V2D/s3out_gpu
rsync -a -e "ssh $SSHO" $SSH_USER@$IP:'~/v2d/s3out/' $SSH_USER@$IP:'~/v2d/run.log' $V2D/s3out_gpu/
ls $V2D/s3out_gpu | grep -c '^episode_......\.npz$'
