#!/bin/bash
#
cd ./models/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint $eachfile
done
cd ../train_test/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done
cd ../unittest/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint $eachfile
done
cd ../utils/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint $eachfile
done
cd ../
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint $eachfile
done