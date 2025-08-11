#!/bin/bash

# Define mount points and remote paths
declare -A mounts=(
    ["fibserver"]="smohinta@fibserver1:/data/raw/"
    ["cephfs"]="smohinta@max:/cephfs/smohinta"
    ["ark"]="smohinta@ark:/data/scratch/smohinta"
)

# Base local mount directory
BASE_DIR="/mnt/scratch/mounts"

# SSHFS options
OPTIONS="-o allow_other"

# Loop through all mounts
for name in "${!mounts[@]}"; do
    remote="${mounts[$name]}"
    local_mount="$BASE_DIR/$name"

    echo "Mounting $remote to $local_mount..."

    # Create local mount point if it doesn't exist
    if [ ! -d "$local_mount" ]; then
        echo "Creating directory $local_mount"
        sudo mkdir -p "$local_mount"
    fi

    # Mount using SSHFS
    sudo sshfs $OPTIONS "$remote" "$local_mount"

    # Check if mount was successful
    if mountpoint -q "$local_mount"; then
        echo "Mounted $name successfully."
    else
        echo "Failed to mount $name."
    fi

    echo
done

