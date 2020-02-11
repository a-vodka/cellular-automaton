#!/bin/bash

filename="vdk_test_vdk_test"

echo "$filename.spectralOut"

postResults --cr f,p --split --separation x,y,z "$filename.spectralOut"
postResults --cr f,p "$filename.spectralOut" --prefix="ttl"

cd ./postProc

addCauchy $filename_inc*.txt
addMises -s Cauchy $filename_inc*.txt

addStrainTensors --left --logarithmic $filename_inc*.txt
addMises -e 'ln(V)' $filename_inc*.txt

#addDisplacement --nodal $filename_inc*.txt


vtk_rectilinearGrid $filename_inc*.txt

for f in *.vtr
do
        fname="${f/_pos(cell)/}"
#       echo $f, $fname
        mv $f $fname
        vtk_addRectilinearGridData --data 'Mises(Cauchy)',1_p,'1_ln(V)',1_Cauchy --vtk $fname "${fname/vtr/txt}"
#        vtk_addRectilinearGridData --data 'fluct(f).pos','avg(f).pos' --vtk $fname "${fname/vtr/txt}"
done

cd ./../

