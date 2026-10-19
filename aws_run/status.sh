#!/bin/bash
cd "$(dirname "$0")"; source common.sh; read ID IP < .instance
ssh $SSHO $SSH_USER@$IP "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls ~/v2d/s3out | grep -c 'npz$'; grep -E '^ep ' ~/v2d/run.log | tail -3"
