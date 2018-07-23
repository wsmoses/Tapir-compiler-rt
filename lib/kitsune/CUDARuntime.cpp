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

#include <iostream>

#include <map>
#include <vector>
#include <cassert>
#include <cmath>
#include <sstream>

#include <cuda.h>

#define np(X)                                                            \
 std::cout << __FILE__ << ":" << __LINE__ << ": " << __PRETTY_FUNCTION__ \
           << ": " << #X << " = " << (X) << std::endl

using namespace std;
using namespace kitsune;

static const uint8_t FIELD_READ = 0x01;
static const uint8_t FIELD_WRITE = 0x02;

static const size_t DEFAULT_BLOCK_SIZE = 128;

static bool _initialized = false;

class CUDARuntime : public GPURuntime{
public:
  static void check(CUresult err){
    if(err != CUDA_SUCCESS){
      const char* s;
      cuGetErrorString(err, &s);
      cerr << "CUDARuntime error: " << s << endl;
      assert(false);
    }
  }

  class CommonField{
  public:
    CommonField(void* hostPtr, CUdeviceptr devPtr, size_t size)
      : hostPtr(hostPtr),
        devPtr(devPtr),
        size(size){

    }

    ~CommonField(){
      CUresult err = cuMemFree(devPtr);
      check(err);
    }

    void* hostPtr;
    CUdeviceptr devPtr;
    size_t size;
  };

  class CommonData{
  public:
    CommonData(){}

    ~CommonData(){
      for(auto& itr : fieldMap_){
        delete itr.second;
      }
    }

    CommonField* getField(const char* name){
      auto itr = fieldMap_.find(name);
      if(itr != fieldMap_.end()){
        return itr->second;
      }

      return nullptr;
    }

    CommonField* addField(const char* name,
                          void* hostPtr,
                          uint32_t elementSize,
                          uint64_t size){

      size *= elementSize;

      CUdeviceptr devPtr;
      CUresult err = cuMemAlloc(&devPtr, size);
      check(err);

      CommonField* field = new CommonField(hostPtr, devPtr, size);
      fieldMap_[name] = field;
      return field;
    }

  private:
    typedef map<const char*, CommonField*> FieldMap_;

    FieldMap_ fieldMap_;
  };

  class Kernel;

  class PTXModule{
  public:    
    PTXModule(const char* ptx){
      CUresult err = cuModuleLoadData(&module_, (void*)ptx);
      check(err);
    }

    Kernel* createKernel(uint32_t kernelId, CommonData* commonData);

  private:
    CUmodule module_;
  };

  class Kernel{
  public:
    class Field{
    public:
      CommonField* commonField;
      uint8_t mode;

      bool isRead(){
        return mode & FIELD_READ;
      }

      bool isWrite(){
        return mode & FIELD_WRITE;
      }
    };

    Kernel(PTXModule* module,
           CommonData* commonData,
           CUfunction function)
      : module_(module),
        commonData_(commonData),
        function_(function),
        ready_(false),
        blockSize_(DEFAULT_BLOCK_SIZE){
      
    }

    ~Kernel(){
      for(auto& itr : fieldMap_){
        delete itr.second;
      }
    }
    
    void setblockSize(size_t blockSize){
      blockSize_ = blockSize;
    }

    void addField(const char* fieldName,
                  CommonField* commonField,
                  uint8_t mode){

      Field* field = new Field;
      field->commonField = commonField;
      field->mode = mode;

      fieldMap_.insert({fieldName, field});
    }

    void run(){
      if(!ready_){
        kernelParams_.push_back(&runSize_);
        kernelParams_.push_back(&runStart_);
        kernelParams_.push_back(&runStart_);
        
        for(auto& itr : fieldMap_){
          Field* field = itr.second;
          CommonField* commonField = field->commonField;
          kernelParams_.push_back(&commonField->devPtr);
        }
        ready_ = true;
      }

      CUresult err;

      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isRead()){
          CommonField* commonField = field->commonField;
          err = cuMemcpyHtoD(commonField->devPtr, commonField->hostPtr,
                             commonField->size);
          check(err);
        }
      }

      assert(runSize_ != 0 && "run size has not been set");

      size_t gridDimX = (runSize_ + blockSize_ - 1)/blockSize_;

      err = cuLaunchKernel(function_, gridDimX, 1, 1,
                           blockSize_, 1, 1, 
                           0, nullptr, kernelParams_.data(), nullptr);
      check(err);

      for(auto& itr : fieldMap_){
        Field* field = itr.second;
        if(field->isWrite()){
          CommonField* commonField = field->commonField;
          err = cuMemcpyDtoH(commonField->hostPtr, commonField->devPtr, 
                             commonField->size);
          check(err);
        }
      }
    }
    
    PTXModule* module(){
      return module_;
    }

    CommonData* commonData(){
      return commonData_;
    }

    bool ready(){
      return ready_;
    }

    void setRunSize(uint64_t size, uint64_t start, uint64_t stride){
      runSize_ = size;
      runStart_ = start;
      runStride_ = stride;
    }
    
  private:    
    typedef map<string, Field*> FieldMap_;
    typedef vector<void*> KernelParams_;

    CUfunction function_;
    PTXModule* module_;
    CommonData* commonData_;
    bool ready_;
    FieldMap_ fieldMap_;
    KernelParams_ kernelParams_;
    size_t blockSize_;
    uint64_t runSize_ = 0;
    uint64_t runStart_ = 0;
    uint64_t runStride_ = 1;
  };

  CUDARuntime(){
    commonData_ = new CommonData;
  }

  ~CUDARuntime(){
    for(auto& itr : kernelMap_){
      delete itr.second;
    }

    for(auto& itr : moduleMap_){
      delete itr.second;
    }

    delete commonData_;

    CUresult err = cuCtxDestroy(context_);
    check(err);
  }

  void init(){
    CUresult err = cuInit(0);
    check(err);
    
    err = cuDeviceGet(&device_, 0);
    check(err);

    err = cuCtxCreate(&context_, 0, device_);
    check(err);

    int threadsPerBlock;
    err = 
      cuDeviceGetAttribute(&threadsPerBlock,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_);
    check(err);
    blockSize_ = threadsPerBlock;
  }

  void initKernel(uint32_t kernelId,
                  const char* ptx){

    auto kitr = kernelMap_.find(kernelId);
    if(kitr != kernelMap_.end()){
      return;
    }

    PTXModule* module;
    auto mitr = moduleMap_.find(ptx);
    if(mitr != moduleMap_.end()){
      module = mitr->second;
    }
    else{
      module = new PTXModule(ptx);
      moduleMap_[ptx] = module;
    }

    Kernel* kernel = module->createKernel(kernelId, commonData_);
    kernel->setblockSize(blockSize_);
    kernelMap_.insert({kernelId, kernel});
  }

  void initField(uint32_t kernelId,
                 const char* fieldName,
                 void* hostPtr,
                 uint32_t elementSize,
                 uint64_t size,
                 uint8_t mode){

    auto kitr = kernelMap_.find(kernelId);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    if(kernel->ready()){
      return;
    }

    CommonData* commonData = kernel->commonData();
    CommonField* commonField = commonData->getField(fieldName);
    if(!commonField){
      commonField = commonData->addField(fieldName, hostPtr,
                                         elementSize, size); 
    }
    
    kernel->addField(fieldName, commonField, mode);
  }

  void runKernel(uint32_t kernelId){
    auto kitr = kernelMap_.find(kernelId);
    assert(kitr != kernelMap_.end() && "invalid kernel");

    Kernel* kernel = kitr->second;
    kernel->run();
  }

  void setRunSize(uint32_t kernelId,
                  uint64_t size,
                  uint64_t start,
                  uint64_t stride) override{
    
    auto itr = kernelMap_.find(kernelId);
    assert(itr != kernelMap_.end() && "invalid kernelId");
    itr->second->setRunSize(size, start, stride);
  }

private:
  typedef map<const char*, PTXModule*> ModuleMap_;
  typedef map<uint32_t, Kernel*> KernelMap_;

  CUdevice device_;
  CUcontext context_;
  size_t blockSize_;

  ModuleMap_ moduleMap_;
  KernelMap_ kernelMap_;
  CommonData* commonData_;
};

CUDARuntime::Kernel* 
CUDARuntime::PTXModule::createKernel(uint32_t kernelId,
                                     CommonData* commonData){
  stringstream kstr;
  kstr << "run" << kernelId;

  CUfunction function;
  CUresult err = cuModuleGetFunction(&function, module_, kstr.str().c_str());
  check(err);
  
  Kernel* kernel = new Kernel(this, commonData, function);
  
  return kernel;
}

extern "C" {

  void __kitsune_cuda_init(){
    if(_initialized){
      return;
    }

    CUDARuntime* runtime = new CUDARuntime;
    runtime->init();

    GPURuntime::init(runtime);

    _initialized = true;
  }

} // extern "C"
