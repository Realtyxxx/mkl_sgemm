#!/bin/bash

echo "##############################################################################################################"
echo "# Title      : test_batch_sgemm                                                                              #"
echo "# CPU        : SKYLAKE                                                                                       #"
echo "# Tester     : TYX                                                                                           #"
echo "# Time       : 2020.3.11                                                                                     #"
echo "# Description: Test performance of mkl 's cblas_sgemm_sgemm for different {groupSize}, {M,N,K_value}.        #"
echo "#              To run the script, one should check environment variables.                                    #"
echo "##############################################################################################################"


echo "------------------------------------------------------------------------ 1 --------------------------------------------------------------------------------"
#### set threads ####
echo "export OMP_NUM_THREADS=1"
echo "export MKL_NUM_THREADS=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#### clean ####

make clean
make rmrecord


#### compile ####
make install
echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------"


echo "------------------------------------------------------------------------ 2 --------------------------------------------------------------------------------"
#### run cblas_gemm_batch ####
# testbatch count M N K

# numbers of matrix
count="8 16 32 64 128 256"

# sizes of a matrix
m=(20 50 500 500 96 48 96 48 64)
k=(25 500 50 64 121 121 49 49 9)
n=(576 64 64 800 3025 3025 12100 12100 50176)
num="0 1 2 3 4 5 6 7 8"

# run
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

for j in $num
do
    echo " "
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    # taskset -c 1 ./cblas_gemm 256 1280
    # taskset -c 1 ./cblas_gemm_batch 256 1280
    # echo 'm-k-n:'${j} >> recordTime.log
    # echo 'm-k-n:'${j} >> recordGflops.log
    # echo 'm-k-n:'${j} >> writeForGflops.log
    echo -e 'm='${m[j]}' k='${k[j]}' n='${n[j]} >> writeBatchGflops.log
    # echo -e 'count:\n8----------8----------16----------16----------32----------32----------64----------64----------128----------128----------256----------256 ' >> recordTime.log
    # echo -e 'count:\n8----------8----------16----------16----------32----------32----------64----------64----------128----------128----------256----------256 ' >> recordGflops.log
    echo -e 'count:\n8----------16----------32----------64----------128----------256 ' >> writeBatchGflops.log
    # echo -e 'count:\n8----------16----------32----------64----------128----------256 ' >> writeForGflops.log
  
    for i in $count
    do
      echo "*************************************************************************************************"
      # export MKL_NUM_THREADS=1 && taskset -c 1 ./batch m k n
      export MKL_NUM_THREADS=1 && taskset -c 1 ./batch $i ${m[j]} ${k[j]} ${n[j]}
      echo -e "export MKL_NUM_THREADS=1 && taskset -c 1 ./batch $i $@{m[j]} $@{k[j]} $@{n[j]}"
    done
    echo -e '' >> writeBatchGflops.log
    echo -e '' >> writeBatchGflops.log
    # echo -e '\n'>>recordTime.log
    # echo -e '\n'>>recordGflops.log
    # echo -e '\n'>>writeBatchGflops.log
    # echo -e '\n'>>writeForGflops.log
    # echo ' '>>record.log
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
done
echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------"
