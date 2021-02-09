#!/bin/bash

# adds current user to grp docker
function addToGrp() {
    grps=$(id -G -n `whoami`)

    # check if already in docker group
    for grp in $grps
    do
        if [ "$grp" == "docker" ]; then
            return 1
        fi
    done
    
    # add to group
    sudo usermod -aG docker `whoami`

}

addToGrp

docker build -t local/handgesture:latest \
    "git@github.com:singalkunal/HandGesture.git#container:docker"
