#!/bin/csh -f

cd /net/brevort/z/guanchen/projects/transformer/nanoGPT_hw/HW/SA

#This ENV is used to avoid overriding current script in next vcselab run 
setenv SNPS_VCSELAB_SCRIPT_NO_OVERRIDE  1

/usr/caen/vcs-2020.12-SP2-1/linux64/bin/vcselab $* \
    -o \
    sim \
    -nobanner \
    +vcs+lic+wait \

cd -

