#!/bin/bash

# adds current user to grp docker
# necessary to run docker commands wo root priviliges
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

# default values
video_source=0
nhands=1
width=600
height=600
image="singalkunal"

function help() {
    printf " -------Usage--------\n"
    printf " Named params: (Optional):\n  -s: video source\n  -n: #hands"
    printf "\n  -w: width\n  -h: height"
    printf "\n Flags:\n  -l: if image built locally using build script"
    printf "\n --------------------\n"
}

while getopts ":s:n:w:h:l" opt; do
    case $opt in
        s) video_source="$OPTARG"
            ;;
        n) nhands="$OPTARG"
            ;;
        w) width="$OPTARG"
            ;;
        h) height="$OPTARG"
            ;;
        l) local_=1
            ;;
        \?) echo "Invalid option -$OPTARG" >&2
            help
            ;;
    esac
done

if [ "$local_" == 1 ]; then
    image="local"
fi

# allows local connections to X server (See man xhost)
# necessary for accessing gui
xhost local:

# enable if errors regarding Xauth
# local +si:localuser:`whoami`

docker run --gpus all \
    --rm \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    "${image}/handgesture:latest" \
    -src "${video_source}" \
    -nhands "${nhands}" \
    -wd "${width}" \
    -ht "${height}"

# reverting xhost settings
xhost -local:
