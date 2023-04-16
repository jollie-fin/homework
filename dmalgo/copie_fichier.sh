#!/bin/sh
for name in `cat hosts`
do
echo $name;
ssh $name cp $1 /tmp;
done

