#ifndef _INPUT_GEN_RUNTIMES_RT_COMMON_H_
#define _INPUT_GEN_RUNTIMES_RT_COMMON_H_

#include <bitset>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>

#include <array>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <vector>

struct RTInfo {};

namespace {
int VERBOSE = 0;
int TIMING = 0;
struct InitGlobalsTy {
  InitGlobalsTy() {
    VERBOSE = (bool)getenv("INPUTGEN_VERBOSE");
    TIMING = (bool)getenv("INPUTGEN_TIMING");
  }
} InitGlobals;
} // namespace

static uint64_t InputGenMagicNumber = 0x4e45475455504e49; // "INPUTGEN"
static uint32_t InputGenVersion = 0;

enum InputMode {
  InputMode_Generate = 0,
  InputMode_Record_v1 = 1,
  InputMode_Record_v2 = 2,
};
static const char *inputModeToStr(enum InputMode Mode) {
  switch (Mode) {
  case InputMode_Generate:
    return "Generate";
  case InputMode_Record_v1:
    return "Record_v1";
  case InputMode_Record_v2:
    return "Record_v2";
  }
  assert(false && "Invalid input mode");
}

#ifndef NDEBUG
#define INPUTGEN_DEBUG(X)                                                      \
  do {                                                                         \
    if (VERBOSE) {                                                             \
      X;                                                                       \
    }                                                                          \
  } while (0)
#else
#define INPUTGEN_DEBUG(X)
#endif

#define INPUTGEN_TIMER_DEFINE(Name)                                            \
  std::chrono::steady_clock::time_point Timer##Name##Begin

#define INPUTGEN_TIMER_START(Name)                                             \
  do {                                                                         \
    if (TIMING)                                                                \
      Timer##Name##Begin = std::chrono::steady_clock::now();                   \
  } while (0)

#define INPUTGEN_TIMER_END(Name)                                               \
  do {                                                                         \
    if (TIMING) {                                                              \
      std::chrono::steady_clock::time_point Timer##Name##End =                 \
          std::chrono::steady_clock::now();                                    \
      std::cout << "Time for " << #Name << ": "                                \
                << std::chrono::duration_cast<std::chrono::nanoseconds>(       \
                       Timer##Name##End - Timer##Name##Begin)                  \
                       .count()                                                \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

static constexpr intptr_t ObjAlignment = 16;
static constexpr intptr_t MaxPrimitiveTypeSize = 16;

static constexpr int UnreachableExitStatus = 111;

typedef uint8_t *VoidPtrTy;
typedef struct {
} *FunctionPtrTy;

template <typename T> static char *ccast(T *Ptr) {
  return reinterpret_cast<char *>(Ptr);
}

template <typename T> static VoidPtrTy toPtr(T Ptr) {
  return reinterpret_cast<VoidPtrTy>(Ptr);
}

static uintptr_t toUint(VoidPtrTy Ptr) {
  return reinterpret_cast<uintptr_t>(Ptr);
}

template <typename T> static void *toVoidPtr(T Ptr) {
  return static_cast<void *>(Ptr);
}

template <unsigned Width, typename T>
static std::bitset<Width> toBitsFixed(T Ptr) {
  return std::bitset<Width>((uintptr_t)Ptr);
}
template <typename T> static std::bitset<sizeof(T) * 8> toBits(T Ptr) {
  return std::bitset<sizeof(T) * 8>((uintptr_t)Ptr);
}

template <typename T> static T readV(std::istream &Input) {
  T El;
  Input.read(ccast(&El), sizeof(El));
  return El;
}

template <typename T> static void writeV(std::ostream &Output, T El) {
  Output.write(ccast(&El), sizeof(El));
}

using OffsetTy = intptr_t;
using SizeTy = uintptr_t;

using MallocFuncTy = void *(*)(size_t);
using FreeFuncTy = void (*)(void *);

struct AllocatorTy {
  MallocFuncTy Malloc;
  FreeFuncTy Free;
};

static std::string getFunctionNameFromFile(std::string FileName,
                                           std::string FuncIdent) {
  std::string OriginalFuncName;
  std::ifstream In(FileName);
  std::string Id;
  while (std::getline(In, Id, '\0') &&
         std::getline(In, OriginalFuncName, '\0') && Id != FuncIdent)
    ;
  if (Id != FuncIdent) {
    std::cerr << "Could not find function with ID " << FuncIdent << " in "
              << FileName << std::endl;
    abort();
  }
  return OriginalFuncName;
}

static void useValue(VoidPtrTy Ptr, uint32_t Size) {
  if (getenv("___INPUT_GEN_USE___"))
    for (unsigned I = 0; I < Size; I++)
      printf("%c\n", *(Ptr + Size));
}

static constexpr intptr_t MinObjAllocation = 64;
static constexpr unsigned NullPtrProbability = 75;
static constexpr int CmpPtrRetryProbability = 10;
static constexpr int MaxDeviationFromBranchHint = 10;

template <typename T> static T divFloor(T A, T B) {
  assert(B > 0);
  T Res = A / B;
  T Rem = A % B;
  if (Rem == 0)
    return Res;
  if (Rem < 0) {
    assert(A < 0);
    return Res - 1;
  }
  assert(A > 0);
  return Res;
}

template <typename T> static T divCeil(T A, T B) {
  assert(B > 0);
  T Res = A / B;
  T Rem = A % B;
  if (Rem == 0)
    return Res;
  if (Rem > 0) {
    assert(A > 0);
    return Res + 1;
  }
  assert(A < 0);
  return Res;
}

template <typename T> static T alignStart(T Ptr, intptr_t Alignment) {
  intptr_t IPtr = reinterpret_cast<intptr_t>(Ptr);
  return reinterpret_cast<T>(divFloor(IPtr, Alignment) * Alignment);
}

template <typename T> static T alignEnd(T Ptr, intptr_t Alignment) {
  intptr_t IPtr = reinterpret_cast<intptr_t>(Ptr);
  return reinterpret_cast<T>(divCeil(IPtr, Alignment) * Alignment);
}

static VoidPtrTy advance(VoidPtrTy Ptr, uint64_t Bytes) {
  return reinterpret_cast<uint8_t *>(Ptr) + Bytes;
}

struct AlignedMemoryChunk {
  VoidPtrTy Ptr;
  intptr_t InputSize;
  intptr_t InputOffset;
  intptr_t OutputSize;
  intptr_t OutputOffset;
  intptr_t CmpSize;
  intptr_t CmpOffset;
};

struct ObjectAddressing {
  virtual size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const = 0;
  virtual OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const = 0;
  virtual OffsetTy getObjBaseOffset() const = 0;
  OffsetTy getOffsetFromObjBasePtr(OffsetTy Ptr) const {
    return Ptr - getObjBaseOffset();
  }
  virtual OffsetTy getLowestObjOffset() const = 0;
  virtual uintptr_t getMaxObjectSize() const = 0;
  virtual ~ObjectAddressing() {};
};

// TODO There should also be a template argument to the class which says whether
// the Input and Used should expand dynamically
struct ObjectTy {
  static constexpr bool StaticSize = false;
  static constexpr bool NeedsToGrowUsedAndInput = !StaticSize;
  ObjectTy(AllocatorTy &Allocator, VoidPtrTy Output, size_t Size,
           OffsetTy Offset, bool AllocateInputUsed,
           bool KnownSizeObjBundle = false)
      : KnownSizeObjBundle(KnownSizeObjBundle), Output(Allocator),
        Input(Allocator), Used(Allocator) {
    INPUTGEN_DEBUG(std::cerr << "Creating Object at #" << this
                             << " with memory " << (void *)Output << " size "
                             << Size << " offset " << Offset << "\n");
    this->Output.Memory = Output;
    this->Output.AllocationSize = Size;
    this->Output.AllocationOffset = Offset;

    if (AllocateInputUsed) {
      this->Input.template ensureAllocation<true>(0, Size);
      // TODO include a mode where we do not track Used on per-byte basis, and
      // only the range
      this->Used.template ensureAllocation<true>(0, Size);
    } else {
      assert(!StaticSize);
    }
    if (KnownSizeObjBundle) {
      CurrentStaticObjEnd = Offset;
    }
  }
  ~ObjectTy() {}

  auto &getPtrs() { return Ptrs; }
  auto &getFPtrs() { return FPtrs; }

  void report() { std::cerr << OutputLimits.getSize() << std::endl; }

  bool KnownSizeObjBundle;
  OffsetTy CurrentStaticObjEnd;

  const auto &getOutputMemory() { return Output; }

  VoidPtrTy getBasePtr() { return Output.Memory - Output.AllocationOffset; }

  OffsetTy getLocalPtr(VoidPtrTy GlobalPtr) { return GlobalPtr - getBasePtr(); }

  VoidPtrTy getGlobalPtr(OffsetTy LocalPtr) { return getBasePtr() + LocalPtr; }

  bool isGlobalPtrInObject(VoidPtrTy GlobalPtr) {
    VoidPtrTy BasePtr = getBasePtr();
    return BasePtr <= GlobalPtr && BasePtr > GlobalPtr;
  }

  OffsetTy getOffsetFromObjMemory(OffsetTy Ptr) {
    return Ptr - Output.AllocationOffset;
  }

  OffsetTy addKnownSizeObject(ObjectAddressing &OA, uintptr_t Size) {
    assert(KnownSizeObjBundle);
    // Make sure zero-sized objects have their own address
    if (Size == 0)
      Size = 1;
    if (Size + CurrentStaticObjEnd >
        OA.getLowestObjOffset() + OA.getMaxObjectSize())
      return 0;
    OffsetTy ObjPtr = CurrentStaticObjEnd;
    CurrentStaticObjEnd =
        alignEnd<intptr_t>(CurrentStaticObjEnd + Size, ObjAlignment);
    return ObjPtr;
  }

  struct KnownSizeObjInputMem {
    OffsetTy Start;
    uintptr_t Size;
  };
  KnownSizeObjInputMem getKnownSizeObjectInputMemory(ObjectAddressing &OA,
                                                     OffsetTy LocalPtr,
                                                     uintptr_t Size) {
    assert(KnownSizeObjBundle);
    KnownSizeObjInputMem Mem;
    Mem.Start = std::min<uintptr_t>(
        LocalPtr + Size,
        std::max<uintptr_t>(LocalPtr,
                            OA.getObjBaseOffset() + InputLimits.LowestOffset));
    OffsetTy End = std::max<uintptr_t>(
        LocalPtr,
        std::min<uintptr_t>(LocalPtr + Size,
                            OA.getObjBaseOffset() + InputLimits.HighestOffset));
    Mem.Size = End - Mem.Start;
    assert(Mem.Start <= End);
    return Mem;
  }

  void comparedAt(VoidPtrTy GlobalPtr) {
    OffsetTy Offset = getLocalPtr(GlobalPtr);
    CmpLimits.update(Offset, 1);
  }

  AlignedMemoryChunk getAlignedInputMemory() {
    // If we compare the pointer at some offset we need to make sure the output
    // allocation will contain those locations, otherwise comparisons may differ
    // in input-gen and input-run as we would compare against an offset in a
    // different object
    if (!OutputLimits.isEmpty()) {
      if (!CmpLimits.isEmpty())
        OutputLimits.update(CmpLimits.LowestOffset, CmpLimits.getSize());
      // We no longer need the CmpLimits, reset it
      CmpLimits = Limits();
    }

    VoidPtrTy InputStart =
        InputLimits.LowestOffset + Input.Memory - Input.AllocationOffset;
    VoidPtrTy InputEnd =
        InputLimits.HighestOffset + Input.Memory - Input.AllocationOffset;
    intptr_t OutputStart = alignStart(OutputLimits.LowestOffset, ObjAlignment);
    intptr_t OutputEnd = alignEnd(OutputLimits.HighestOffset, ObjAlignment);
    return {InputStart,
            InputEnd - InputStart,
            InputLimits.LowestOffset,
            OutputEnd - OutputStart,
            OutputStart,
            CmpLimits.getSize(),
            CmpLimits.LowestOffset};
  }

  template <typename T> T read(OffsetTy Ptr, uint32_t Size, RTInfo *RTI);
  template <typename T>
  T readFromGlobalPtr(VoidPtrTy Ptr, uint32_t Size, RTInfo *RTI) {
    return read<T>(getLocalPtr(Ptr), Size, RTI);
  }

  template <typename T> void write(T Val, OffsetTy Ptr, uint32_t Size) {
    intptr_t Offset = getOffsetFromObjMemory(Ptr);
    assert(Output.isAllocated(Offset, Size));
    Used.ensureAllocation(Offset, Size);
    markUsed(Offset, Size);
    OutputLimits.update(Offset, Size);
  }
  template <typename T>
  void writeToGlobalPtr(T Val, VoidPtrTy Ptr, uint32_t Size) {
    write<T>(Val, getLocalPtr(Ptr), Size);
  }

  void setFunctionPtrIdx(OffsetTy Ptr, uint32_t Size, VoidPtrTy FPtr,
                         uint32_t FIdx) {
    intptr_t Offset = getOffsetFromObjMemory(Ptr);
    storeInputValue(FPtr, Offset, Size);
    FPtrs.insert({Offset, FIdx});
  }

  std::set<intptr_t> Ptrs;
  std::unordered_map<intptr_t, uint32_t> FPtrs;

public:
  template <bool IsBitSet = false> struct MemoryTy {
    AllocatorTy &Allocator;
    MemoryTy(AllocatorTy &Allocator) : Allocator(Allocator) {}
    VoidPtrTy Memory = nullptr;
    intptr_t AllocationSize = 0;
    OffsetTy AllocationOffset = 0;
    bool isAllocated(intptr_t Offset, uint32_t Size) {
      intptr_t AllocatedMemoryStartOffset = AllocationOffset;
      intptr_t AllocatedMemoryEndOffset =
          AllocatedMemoryStartOffset + AllocationSize;
      return (AllocatedMemoryStartOffset <= Offset &&
              AllocatedMemoryEndOffset >= Offset + Size);
    }

    /// Returns true if it was already allocated
    template <bool Exact = false>
    bool ensureAllocation(intptr_t Offset, uint32_t Size) {
      if (isAllocated(Offset, Size))
        return true;
      reallocateData<Exact>(Offset, Size);
      return false;
    }

    void setBit(size_t Idx) {
      if constexpr (IsBitSet) {
        Memory[Idx / 8] = Memory[Idx / 8] | (1 << (Idx % 8));
      } else {
        assert(0);
      }
    }
    uint8_t operator[](size_t Idx) {
      if constexpr (IsBitSet) {
        uint8_t Byte = Memory[Idx / 8];
        return (Byte >> (Idx % 8)) & 1;
      } else {
        return Memory[Idx];
      }
    }

    template <typename T>
    void extendMemory(T *&OldMemory, intptr_t NewAllocationSize,
                      intptr_t NewAllocationOffset) {
      if constexpr (IsBitSet) {
        assert(NewAllocationSize % 8 == 0);
        static_assert(ObjAlignment % 8 == 0);
        NewAllocationSize = NewAllocationSize / 8;
      }
      T *NewMemory = reinterpret_cast<T *>(Allocator.Malloc(NewAllocationSize));
      assert(NewMemory && "Malloc failed");
      memset(NewMemory, 0, NewAllocationSize);
      memcpy(advance(NewMemory, AllocationOffset - NewAllocationOffset),
             OldMemory, AllocationSize);
      Allocator.Free(OldMemory);
      OldMemory = NewMemory;
    };

    /// Reallocates the data so as to make the memory at `Offset` with length
    /// `Size` available
    template <bool Exact = false>
    void reallocateData(intptr_t Offset, uint32_t Size) {
      assert(!isAllocated(Offset, Size));

      intptr_t AllocatedMemoryStartOffset = AllocationOffset;
      intptr_t AllocatedMemoryEndOffset =
          AllocatedMemoryStartOffset + AllocationSize;
      intptr_t NewAllocatedMemoryStartOffset = AllocatedMemoryStartOffset;
      intptr_t NewAllocatedMemoryEndOffset = AllocatedMemoryEndOffset;

      intptr_t AccessStartOffset = Offset;
      intptr_t AccessEndOffset = AccessStartOffset + Size;

      if (AccessStartOffset < AllocatedMemoryStartOffset) {
        // Extend the allocation in the negative direction
        if constexpr (Exact) {
          NewAllocatedMemoryStartOffset =
              alignStart(AccessStartOffset, ObjAlignment);
        } else {
          NewAllocatedMemoryStartOffset = alignStart(
              std::min(2 * AccessStartOffset, -MinObjAllocation), ObjAlignment);
        }
      }
      if (AccessEndOffset >= AllocatedMemoryEndOffset) {
        // Extend the allocation in the positive direction
        if constexpr (Exact) {
          NewAllocatedMemoryEndOffset = alignEnd(AccessEndOffset, ObjAlignment);
        } else {
          NewAllocatedMemoryEndOffset = alignEnd(
              std::max(2 * AccessEndOffset, MinObjAllocation), ObjAlignment);
        }
      }

      intptr_t NewAllocationOffset = NewAllocatedMemoryStartOffset;
      intptr_t NewAllocationSize =
          NewAllocatedMemoryEndOffset - NewAllocatedMemoryStartOffset;

      INPUTGEN_DEBUG(
          printf("Reallocating data in Object for access at %ld with size %d "
                 "from offset "
                 "%ld, size %ld to offset %ld, size %ld.\n",
                 Offset, Size, AllocationOffset, AllocationSize,
                 NewAllocationOffset, NewAllocationSize));

      extendMemory(Memory, NewAllocationSize, NewAllocationOffset);

      AllocationSize = NewAllocationSize;
      AllocationOffset = NewAllocationOffset;
    }
  };

private:
  MemoryTy<false> Output, Input;
  MemoryTy<true> Used;

  struct Limits {
    bool Initialized = false;
    intptr_t LowestOffset = 0;
    intptr_t HighestOffset = 0;
    bool isEmpty() { return !Initialized; }
    intptr_t getSize() { return HighestOffset - LowestOffset; }
    void update(intptr_t Offset, uint32_t Size) {
      if (!Initialized) {
        Initialized = true;
        LowestOffset = Offset;
        HighestOffset = Offset + Size;
        return;
      }
      if (LowestOffset > Offset)
        LowestOffset = Offset;
      if (HighestOffset < Offset + Size)
        HighestOffset = Offset + Size;
    }
  };
  Limits InputLimits, OutputLimits, CmpLimits;

  bool allUsed(intptr_t Offset, uint32_t Size) {
    for (unsigned It = 0; It < Size; It++)
      if (!Used.isAllocated(Offset + It, 1) ||
          !Used[Offset + It - Used.AllocationOffset])
        return false;
    return true;
  }

  void markUsed(intptr_t Offset, uint32_t Size) {
    assert(Used.isAllocated(Offset, Size));

    for (unsigned It = 0; It < Size; It++)
      Used.setBit(Offset + It - Used.AllocationOffset);
  }

  template <typename T, bool StoerOutput = true>
  void storeInputValue(T Val, intptr_t Offset, uint32_t Size) {
    assert(Size == sizeof(Val));

    // Only assign the bytes that were uninitialized
    uint8_t Bytes[sizeof(Val)];
    memcpy(Bytes, &Val, sizeof(Val));
    for (unsigned It = 0; It < sizeof(Val); It++) {
      if (!allUsed(Offset + It, 1)) {
        VoidPtrTy OutputLoc =
            Output.Memory - Output.AllocationOffset + Offset + It;
        VoidPtrTy InputLoc =
            Input.Memory - Input.AllocationOffset + Offset + It;
        if constexpr (StoerOutput)
          *OutputLoc = Bytes[It];
        *InputLoc = Bytes[It];
        markUsed(Offset + It, 1);
      }
    }

    // TODO think about whether we need this for input recording.
    InputLimits.update(Offset, Size);
    OutputLimits.update(Offset, Size);
  }
};

template <typename RTTy>
struct InputRecordObjectAddressing : public ObjectAddressing {
  OffsetTy getObjBaseOffset() const override { return 0; }
  OffsetTy getLowestObjOffset() const override { abort(); }
  uintptr_t getMaxObjectSize() const override { abort(); }

  RTTy &RT;
  InputRecordObjectAddressing(RTTy &RT) : RT(RT) {}

  OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const override {
    auto Res = RT.globalPtrToObjAndLocalPtr(GlobalPtr);
    assert(Res);
    return Res->second;
  }
  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, OffsetTy PtrInObj) const {
    return RT.localPtrToGlobalPtr(ObjIdx, PtrInObj);
  }

  size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const override { abort(); }

  uintptr_t getSize() { abort(); };
};

template <size_t N> static constexpr size_t getStorageForSize() {
  if constexpr (N == 1) {
    return 1;
  } else if constexpr (N == 2) {
    return 2;
  } else if constexpr (N == 3) {
    return 2;
  } else {
    static_assert(0);
    return 0;
  }
}

struct MemoryChunkTy {
  VoidPtrTy Location;
  VoidPtrTy Memory;
  size_t Size;
};

struct RecordDumpObjectTy {
  MemoryChunkTy &Input;

  RecordDumpObjectTy(MemoryChunkTy &I) : Input(I) {}

  std::vector<intptr_t> Ptrs;
  std::unordered_map<intptr_t, uint32_t> FPtrs;
  auto &getPtrs() { return Ptrs; }
  auto &getFPtrs() { return FPtrs; }

  bool isGlobalPtrInObject(VoidPtrTy GlobalPtr) {
    // FIXME  terrible performance
    VoidPtrTy BasePtr = getBasePtr();
    return BasePtr <= GlobalPtr && BasePtr + Input.Size > GlobalPtr;
  }
  VoidPtrTy getBasePtr() { return Input.Location; }
  OffsetTy getLocalPtr(VoidPtrTy GlobalPtr) { return GlobalPtr - getBasePtr(); }
  VoidPtrTy getGlobalPtr(OffsetTy LocalPtr) { return getBasePtr() + LocalPtr; }

  VoidPtrTy getOutputMemoryPtr() { return Input.Location; }
  size_t getOutputMemorySize() { return Input.Size; }
  AlignedMemoryChunk getAlignedInputMemory() {
    return {
        Input.Memory, (intptr_t)Input.Size, 0, (intptr_t)Input.Size, 0, 0, 0};
  }
};

struct InputGenObjectAddressing : public ObjectAddressing {
  ~InputGenObjectAddressing(){};
  size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const override {
    size_t Idx =
        (reinterpret_cast<intptr_t>(GlobalPtr) & ObjIdxMask) / MaxObjectSize;
    return Idx - ObjIdxOffset;
  }

  OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const override {
    return reinterpret_cast<intptr_t>(GlobalPtr) & PtrInObjMask;
  }

  OffsetTy getObjBaseOffset() const override { return MaxObjectSize / 2; }

  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, OffsetTy PtrInObj) const {
    return reinterpret_cast<VoidPtrTy>(
        ((ObjIdxOffset + ObjIdx) * MaxObjectSize) |
        reinterpret_cast<intptr_t>(PtrInObj));
  }

  uintptr_t getMaxObjectSize() const override { return MaxObjectSize; }
  OffsetTy getLowestObjOffset() const override { return 0; }
  uintptr_t getSize() { return Size; };
  uintptr_t getMaxObjectNum() { return MaxObjectNum; }
  uintptr_t getMaxObjectSize() { return MaxObjectSize; }

  void setObjIdxOffset(uintptr_t Offset) { ObjIdxOffset = Offset; }

  void setSize(uintptr_t Size) {
    this->Size = Size;

    uintptr_t HO = highestOne(Size | 1);
    uintptr_t BitsForObj = HO * 70 / 100;
    uintptr_t BitsForObjIndexing = HO - BitsForObj;
    MaxObjectSize = 1ULL << BitsForObj;
    MaxObjectNum = 1ULL << (BitsForObjIndexing);
    PtrInObjMask = MaxObjectSize - 1;
    ObjIdxMask = ~(PtrInObjMask);
    INPUTGEN_DEBUG(std::cerr << "OA " << BitsForObj
                             << " bits for in-object addressing and "
                             << BitsForObjIndexing << " for object indexing\n");
  }

private:
  intptr_t PtrInObjMask;
  intptr_t ObjIdxMask;
  uintptr_t MaxObjectSize;
  uintptr_t MaxObjectNum;

  uintptr_t ObjIdxOffset = 0;
  uintptr_t Size = 0;

  unsigned int highestOne(uint64_t X) { return 63 ^ __builtin_clzll(X); }
};

struct GenValTy {
  uint8_t Content[MaxPrimitiveTypeSize] = {0};
  static_assert(sizeof(Content) == MaxPrimitiveTypeSize);
  int32_t IsPtr;
};

template <typename T> static GenValTy toGenValTy(T A, int32_t IsPtr) {
  GenValTy U;
  static_assert(sizeof(T) <= sizeof(U.Content));
  memcpy(U.Content, &A, sizeof(A));
  U.IsPtr = IsPtr;
  return U;
}

#endif // _INPUT_GEN_RUNTIMES_RT_COMMON_H_
