#!/bin/bash

ssh-keygen -q -t dsa -f ~/.ssh/id_dsa_tp_mpi -N ""
cat ~/.ssh/id_dsa_tp_mpi.pub >> ~/.ssh/authorized_keys
cat known_hosts.europe >> ~/.ssh/known_hosts

