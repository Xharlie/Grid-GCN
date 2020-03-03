#mx_home=/root/dev/mxnet-apache
mx_home=/home/xharlie/dev/mxnet-apache
dmlc_inc=${mx_home}/3rdparty/dmlc-core/include
nnvm_inc=${mx_home}/3rdparty/nnvm/include
mshadow_inc=${mx_home}/3rdparty/mshadow
dlpack_inc=${mx_home}/3rdparty/dlpack/include
mxnet_src_inc=${mx_home}/src/operator
mxnet_inc=${mx_home}/include
cuda_inc=/usr/local/cuda/include

mxnet_lib = ${mx_home}
INC = -I${dmlc_inc} -I${nnvm_inc} -I${mxnet_inc} -I${mshadow_inc} -I${dlpack_inc} -I${cuda_inc} -I${mxnet_src_inc} 
DEFINE = -D MSHADOW_USE_CBLAS -D MSHADOW_USE_CUDA -D MSHADOW_USE_CUDNN=1 -D MSHADOW_USE_CUSOLVER=1 -DMSHADOW_USE_F16C=0
LIB = -L${mxnet_lib}/lib -lmxnet -L/usr/local/cuda/lib64/ -lcudadevrt

all: clean build cleancache

build: additional.so

# 30 35 50 52 60 61 70

%.o: %.cu
	nvcc -std=c++11 -c -o $@ $? -O2 -x cu -Xcompiler -fPIC ${INC} ${DEFINE} -arch=sm_61  -lcudadevrt
#-rdc=true
#
#gridify_l.o: gridify.o
#	nvcc -arch=sm_61 -Xcompiler -fPIC -dlink $? -o $@  -lcudadevrt -lcudart
#
#gridify_up_l.o: gridify_up.o
#	nvcc -arch=sm_61 -Xcompiler -fPIC -dlink $? -o $@  -lcudadevrt -lcudart

additional.so: $(wildcard *.cc) $(subst .cu,.o, $(wildcard *.cu)) #gridify_l.o gridify_up_l.o
	g++ -std=c++11 -shared -o $@ $?  -O2 -msse2 -msse3 -fPIC ${INC} ${DEFINE} ${LIB}

clean:
	rm -f *.o *.so *.pyc

cleancache:
	rm -f *.o

