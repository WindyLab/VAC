#!/bin/bash

# initialize conda
HOSTNAME=$(hostname)
CONDA_PATH="/home/$HOSTNAME/anaconda3"

__conda_setup="$('$CONDA_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_PATH/bin:$PATH"
    fi
fi
unset __conda_setup
(
    conda deactivate
    roscore
) &
(
    sleep 1
    conda deactivate
    cd vas_ws
    source ./devel/setup.bash
    rosrun vas image_concatenator _host_id:=1 2>&1 >/dev/null </dev/null
) &
(
    sleep 1
    conda activate vas
    python3 ros_nodes/ob_detection.py 2>&1 >/dev/null </dev/null

) &
(
    sleep 1
    conda activate vas
    python3 ros_nodes/onboard_v_attention.py 2>&1 >/dev/null </dev/null
) &
(
    sleep 1
    conda deactivate
    cd vas_ws
    source ./devel/setup.bash
    rosrun vas plkf_control 2>&1 >/dev/null </dev/null
)
