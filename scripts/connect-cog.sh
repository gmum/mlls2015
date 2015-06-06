#!/bin/sh

sudo umount -l cog
sudo umount -l cog_data

sshfs sjastrzebski@ailab.ii.uj.edu.pl:/home/sjastrzebski/al_ecml cog
sshfs sjastrzebski@ailab.ii.uj.edu.pl:/home/sjastrzebski/al_ecml_big cog_data

