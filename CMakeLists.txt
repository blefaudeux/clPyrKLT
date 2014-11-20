cmake_minimum_required (VERSION 2.6) 
project (clPyrKLT) 

find_package(OpenCV REQUIRED)

# Recurse into the "libCudaLK" and "demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure.
add_subdirectory (CudaLK)
add_subdirectory (Demo) 
