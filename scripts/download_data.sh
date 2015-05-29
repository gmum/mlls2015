#!/bin/bash

wget -H -r --level=1 -k -p http://pasieka.ii.uj.edu.pl/datasets/farmakologia/
mv pasieka.ii.uj.edu.pl/datasets/farmakologia/*.libsvm .
rm -r pasieka.ii.uj.edu.pl/


