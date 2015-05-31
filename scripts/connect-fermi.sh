#!/bin/sh
sudo umount -l fermi
sudo umount -l fermi-data
sshfs czarnecki@fermi.ii.uj.edu.pl:/lhome/home/czarnecki/st/al_ecml ./fermi
sshfs czarnecki@fermi.ii.uj.edu.pl:/mnt/users/czarnecki/local/al_ecml ./fermi-data

