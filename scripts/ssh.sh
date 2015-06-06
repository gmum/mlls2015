#!/bin/bash

if [ "$1" == "msi" ]; then
    ssh -t user@192.168.0.14 -p 5700 "cd /home/user/staszek/al_ecml; bash"
fi

if [ "$1" == "rob" ]; then
    ssh -t sjastrzebski@149.156.65.220 "cd /home/sjastrzebski/al_ecml; bash"
fi

if [ "$1" == "cog" ]; then
    ssh -t sjastrzebski@cogito.ii.uj.edu.pl "cd /home/sjastrzebski/al_ecml; bash"
fi

if [ "$1" == "hub" ]; then
    ssh -t staszek.jastrzebski@104.155.96.215 "cd /home/staszek.jastrzebski/al_ecml; bash"
fi

if [ "$1" == "fermi" ]; then
    ssh -t czarnecki@fermi.ii.uj.edu.pl "cd /lhome/home/czarnecki/st/al_ecml; bash"
fi
