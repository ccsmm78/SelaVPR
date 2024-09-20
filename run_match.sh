#!/bin/bash
. ~/.virtualenvs/gd_venv/bin/activate

python match.py --datasets_folder=./ --resume=/home/ccsmm/DB_Repo/pretrained/SelaVPR/SelaVPR_reg4_msls.pth --registers
