#!/bin/bash
/usr/local/cuda/bin/ncu --print-source cuda,sass --import-source 1 --page source -o ncu_report%i --set full ./a.out