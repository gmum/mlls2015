#!/bin/sh

sudo umount -l hub
sudo umount -l hub_data

sshfs staszek.jastrzebski@104.155.96.215:/home/staszek.jastrzebski/al_ecml hub
sshfs staszek.jastrzebski@104.155.96.215:/home/staszek.jastrzebski/al_ecml-big hub_data

