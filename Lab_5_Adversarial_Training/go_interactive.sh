#!/bin/bash
module add libs/tensorflow/1.2

# Inside lab:
# srun -p gpu --gres=gpu:1 -A comsm0018 --reservation=comsm0018-lab3  -t 0-02:00 --mem=4G --pty bash

# Outside lab:
srun -p gpu --gres=gpu:1 -t 0-02:00 --mem=4G --pty bash
