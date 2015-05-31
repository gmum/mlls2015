#!/bin/sh

rsync -arv --no-perms --no-owner --no-group  -u --exclude-from 'exclude.rsync' ./* $1


