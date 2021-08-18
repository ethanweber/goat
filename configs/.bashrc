cudaset() {
    export CUDA_HOME="/usr/local/$1"
    export PATH="/usr/local/$1/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/$1/lib64:$LD_LIBRARY_PATH"
    $CUDA_HOME/bin/nvcc --version
}

cudavis() {
    echo "export CUDA_VISIBLE_DEVICES=$1"
    export CUDA_VISIBLE_DEVICES="$1"
    echo "echo \$CUDA_VISIBLE_DEVICES"
    echo "$CUDA_VISIBLE_DEVICES"
}
