if [ "$2" == "" ]; then
    ssh -N -f -L localhost:$1:localhost:8888 czarnecki@fermi.ii.uj.edu.pl
fi

if [ "$2" == "fermi" ]; then
    ssh -N -f -L localhost:$1:localhost:8888 czarnecki@fermi.ii.uj.edu.pl
fi

if [ "$2" == "cog" ]; then
    ssh -N -f -L localhost:$1:localhost:8888 sjastrzebski@cogito.ii.uj.edu.pl
fi

if [ "$2" == "hub" ]; then
    ssh -N -f -L localhost:$1:localhost:8888 staszek.jastrzebski@104.155.96.215
fi


