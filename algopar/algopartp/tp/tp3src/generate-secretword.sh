#!/bin/sh

if [ ! -f $HOME/.mpd.conf ]; then
    cd $HOME
    touch .mpd.conf
    chmod 600 .mpd.conf
    SW=`hostname`$USER
    echo "secretword=$SW" > .mpd.conf
    cd -
fi
