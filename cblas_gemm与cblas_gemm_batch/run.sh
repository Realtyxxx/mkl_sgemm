#!/bin/bash

echo "##############################################################################################################"
echo "# Title      : test_batch_sgemm                                                                              #"
echo "# CPU        : INTEL I5 8500                                                                                 #"
echo "# Tester     : TYX                                                                                           #"
echo "# Time       : 2020.12.13                                                                                    #"
echo "# Description: Test performance of batch_sgemm for different {groupSize}, {M,N,K_value}.                     #"
echo "#              To run the script, one should check environment variables.                                    #"
echo "##############################################################################################################"


echo "---------------------------------------------------------- 1 ----------------------------------------------------------------"
#### set threads ####
# echo "export OMP_NUM_THREADS=1"
# export OMP_NUM_THREADS=1

#### clean ####
# echo "rm testbatch"
# rm testbatch
make clean
make rmrecord


#### compile ####
echo "g++ *cc -o * -lmkl_r"
make install
echo "-----------------------------------------------------------------------------------------------------------------------------"


echo "---------------------------------------------------------- 2 ----------------------------------------------------------------"
#### run cblas_gemm_batch ####
# testbatch count M N K

# numbers of matrix
count="8 16 32 64 128 256"
# count="1"
# sizes of a matrix
m="40 80 160 320 640 1280"
#n="10 20 40 80 160 320 640 1280" 
#k="10 20 40 80 160 320 640 1280"
#m="3"
#n="3"
#k="3"

#p=(R R R R C C C C)
#q=(R C R C C R C R)
#r=(R R C C C C R R)
# run
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

for j in $m
do
    echo " "
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    # taskset -c 1 ./cblas_gemm 256 1280
    # taskset -c 1 ./cblas_gemm_batch 256 1280
    echo 'm-k-n:'${j} >> recordTime.log
    echo 'm-k-n:'${j} >> recordGflops.log
    echo 'm-k-n:'${j} >> writeForGflops.log
    echo 'm-k-n:'${j} >> writeBatchGflops.log
    echo -e 'count:\n8--------8--------16--------16--------32--------32--------64--------64--------128--------128--------256--------256 ' >> recordTime.log
    echo -e 'count:\n8--------8--------16--------16--------32--------32--------64--------64--------128--------128--------256--------256 ' >> recordGflops.log
    echo -e 'count:\n8--------16--------32--------64--------128--------256 ' >> writeBatchGflops.log
    echo -e 'count:\n8--------16--------32--------64--------128--------256 ' >> writeForGflops.log
  
    for i in $count
    do
      echo "*************************************************************************************************"
      taskset -c 1 ./cblas_gemm ${i} ${j}
      taskset -c 1 ./cblas_gemm_batch ${i} ${j}
      #  echo "./testbatch count=${i} m=${j} m=${j} k=${j}"
    #		for t in {0..7};
    #		do
    #		    echo "------count = ${i}-------count=${i} m=${j} m=${j} k=${j}------layoutA=${p[$t]} layoutB=${q[$t]} layoutC=${r[$t]}-------------------------------------------------------------"
    #        taskset -c 1 ./testbatch $i $j $j $j ${p[$t]} ${q[$t]} ${r[$t]}
    #			echo "----------------------------------------------------------------------------------------------------------------------------"
    #		done	
    #    echo "********************************************************************************************************************"
    done
    echo -e '\n'>>recordTime.log
    echo -e '\n'>>recordGflops.log
    echo -e '\n'>>writeBatchGflops.log
    echo -e '\n'>>writeForGflops.log
    # echo ' '>>record.log
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
done
echo "-----------------------------------------------------------------------------------------------------------------------------"
