/**
  ***************************************************************************
  * Copyright (c) 2017, Los Alamos National Security, LLC.
  * All rights reserved.
  *
  *  Copyright 2010. Los Alamos National Security, LLC. This software was
  *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
  *  Alamos National Laboratory (LANL), which is operated by Los Alamos
  *  National Security, LLC for the U.S. Department of Energy. The
  *  U.S. Government has rights to use, reproduce, and distribute this
  *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
  *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
  *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  *  derivative works, such modified software should be clearly marked,
  *  so as not to confuse it with the version available from LANL.
  *
  *  Additionally, redistribution and use in source and binary forms,
  *  with or without modification, are permitted provided that the
  *  following conditions are met:
  *
  *    * Redistributions of source code must retain the above copyright
  *      notice, this list of conditions and the following disclaimer.
  *
  *    * Redistributions in binary form must reproduce the above
  *      copyright notice, this list of conditions and the following
  *      disclaimer in the documentation and/or other materials provided
  *      with the distribution.
  *
  *    * Neither the name of Los Alamos National Security, LLC, Los
  *      Alamos National Laboratory, LANL, the U.S. Government, nor the
  *      names of its contributors may be used to endorse or promote
  *      products derived from this software without specific prior
  *      written permission.
  *
  *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
  *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
  *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
  *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
  *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
  *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
  *  SUCH DAMAGE.
  *
  ***************************************************************************/


#include "kitsune/GPURuntime.h"

#include <cassert>

using namespace std;
using namespace kitsune;

static GPURuntime* _instance = nullptr;

GPURuntime* GPURuntime::get(){
  assert(_instance && "GPU runtime has not been initialized");
  return _instance;
}

void GPURuntime::finish(){
  if(_instance){
    delete _instance;
    _instance = nullptr;
  }
}

void GPURuntime::init(GPURuntime* runtime){
  assert(!_instance && "GPU runtime has already been initialized");
  _instance = runtime;
}

extern "C" {

  void __kitsune_gpu_finish(){
    GPURuntime::finish();
  }

  void __kitsune_gpu_init_kernel(uint32_t kernelId,
                                 const char* data){
    GPURuntime* runtime = GPURuntime::get();
    runtime->initKernel(kernelId, data);
  }

  void __kitsune_gpu_init_field(uint32_t kernelId,
                                const char* fieldName,
                                void* hostPtr,
                                uint32_t elementSize,
                                uint64_t size,
                                uint8_t mode){
    GPURuntime* runtime = GPURuntime::get();
    runtime->initField(kernelId, fieldName, hostPtr,
                       elementSize, size, mode);
  }

  void __kitsune_gpu_run_kernel(uint32_t kernelId){
    GPURuntime* runtime = GPURuntime::get();
    runtime->runKernel(kernelId);
  }

  void __kitsune_gpu_set_run_size(uint32_t kernelId,
                                  uint64_t size,
                                  uint64_t start,
                                  uint64_t stride){
    GPURuntime* runtime = GPURuntime::get();
    runtime->setRunSize(kernelId, size, start, stride);
  }
  
} // extern "C"
