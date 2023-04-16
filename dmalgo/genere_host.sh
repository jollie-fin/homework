#!/bin/sh
echo -n "" > hosts
for ((j = 0; j <= 2; j+=1))
do
for ((i = 0; i <= 9 ; i += 1))
do
A="slsu"$j"-0"$i
ssh $A exit
if (($? == 0)); then
  echo "slsu"$j"-0"$i >> hosts
fi
done
for ((i = 10; i <= 29 ; i += 1))
do
A="slsu"$j"-"$i
ssh $A exit
if (($? == 0)); then
  echo "slsu"$j"-"$i >> hosts
fi
done
done
