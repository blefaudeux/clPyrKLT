#ifndef LIBCLKLT_H
#define LIBCLKLT_H

#include <iostream>
#include <CL/cl.hpp>

class LibClKLT  {
  private:
    bool allocate_memory();
    bool release_memory();

  public:
    LibClKLT();


};

#endif // LIBCLKLT_H
