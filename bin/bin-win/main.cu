// Style: http://geosoft.no/development/cppstyle.html

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cppformat/format.h>

#include "cuda_helpers.h"
#include "int_types.h"

using namespace std;
using namespace fmt;

// 8 operations
const u32 nOps = 9;
const char* opStrArr[] = {"nop", "v1 ", "v2 ", "v3 ", "v4 ", "and", "or ", "eor", "not"};
const u32 NOP = 0, V1 = 1, V2 = 2, V3 = 3, V4 = 4, AND = 5, OR = 6, EOR = 7, NOT = 8;

const u32 nSearchLevelsBase = 8; //8;
const u32 nSearchLevelsThread = 17;//16;
const u32 maxStackItems = nSearchLevelsThread / 2 + 1;
const u32 nTotalTruthTables = 1 << 16;

typedef u16 Bits;
typedef Bits BitsStack[maxStackItems];
struct Frame {
    u32 op;
    BitsStack stack;
    u32 nStackItems;
};
typedef Frame FrameArr[nSearchLevelsThread];
struct Program {
    u32 opArr[nSearchLevelsThread];
    u32 nOps;
};
typedef Program ProgramArr[nTotalTruthTables];
struct Base {
    Program program;
    Frame frame;
};
typedef vector<Base> BaseVec;
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

__device__ __managed__ ProgramArr optimalProgramArr;
__device__ __managed__ u32 nFilledTruthTables = 0;
__device__ __managed__ u64 nValidProgramsFound = 0;

__device__ __managed__ u64 nFoundPrograms[nTotalTruthTables];

bool initCuda(u32 cudaDevice);
BaseVec generateBases();
void searchBase();
void gpuSearch(const BaseVec& baseVec);
__global__ void searchKernel(u32 nThreads, const Base* baseVec);
void cpuCheckBase(const Base& base);
__device__ void gpuCheckBase(const Base& base);
void printLastEvaluated(const Base& base, const FrameArr& frameArr, u32 nCurrentLevel, Bits truthTable);
void printStatus();
void cpuSearch(const BaseVec& baseVec);
void gpuSearch(const BaseVec& baseVec);
inline void testProgramGenerator();
__host__ __device__ inline u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels, u32 nSearchLevels, bool makeBases);
__host__ __device__ inline void evalOperation(Frame&);
__host__ __device__ inline void push(Frame& f, Bits v);
__host__ __device__ inline Bits pop(Frame& s);
void writeResults(const string& path, const ProgramArr& optimalProgramArr);
void writeHistogram(const string& path);
string serializeProgram(const FrameArr&, u32 nFrames);
string serializeBase(const Base& base);
string serializeFrame(const Frame& f);
string secondsToHms(double sec);

auto startTime = Time::now();

int main(int argc, char *argv[])
{
    // Switch from C locale to user's locale. This will typically cause integers to be printed with thousands
    // separators.
    locale::global(locale(""));
    cout.imbue(locale(""));

    print("Search levels: {}\n", nSearchLevelsThread);
    print("Base levels: {}\n", nSearchLevelsBase);
    print("Max stack items: {}\n", maxStackItems);

    // testProgramGenerator();
    // return 0;

    memset(optimalProgramArr, 0, sizeof(optimalProgramArr));
    memset(nFoundPrograms, 0, sizeof(nFoundPrograms));

    auto baseVec = generateBases();
    print("Bases: {}\n", baseVec.size());

////    int i = 0;
//    for (auto base : baseVec) {
////        if (i++ == 32) {
////            break;
////        }
//        print("{:<50}{}\n", "Sorted base", serializeBase(base));
//    }

    // Quick search to find the programs that have length lower or equal to the base.
    searchBase();

    if (argc == 2) {
        u32 cudaDevice = atoi(argv[1]);
        if (!initCuda(cudaDevice)) {
            return 1;
        }
        print("\nRunning GPU search\n");
        gpuSearch(baseVec);
        writeHistogram(format("bitwise-hist-{}-{}-{}.txt", nSearchLevelsBase, nSearchLevelsThread, nValidProgramsFound));
    }
    else {
        print("\nCUDA device not selected. Running CPU search\n");
        cpuSearch(baseVec);
    }

    printStatus();
    writeResults("bitwise.txt", optimalProgramArr);
    return 0;
}

bool initCuda(u32 cudaDevice)
{
	u32 cudaDeviceCount;
	checkCudaErrorsNoSync(cudaGetDeviceCount(reinterpret_cast<int*>(&cudaDeviceCount)));
	if (!cudaDeviceCount) {
		print("Error: Found no devices supporting CUDA\n");
        return false;
	}

    print("\nUsing CUDA device: {0}\n\n", cudaDevice);
    checkCudaErrors(cudaSetDevice(cudaDevice));
    printDeviceInfo(cudaDevice);
    return true;
}

// CPU

void cpuSearch(const BaseVec& baseVec)
{
    for (auto base : baseVec) {
        cpuCheckBase(base);
    }
}

void cpuCheckBase(const Base& base)
{
    u32 nCurrentLevel = 0;
    u32 nSearchLevels = nSearchLevelsThread - nSearchLevelsBase + 1 /* nop */;
    Frame frameArr[nSearchLevelsThread];
//    memcpy(frameArr, &base.frame, sizeof(base.frame));
    frameArr[0] = base.frame;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1, nSearchLevels, false))) {
        ++nValidProgramsFound;
        Bits truthTable = frameArr[nCurrentLevel].stack[0];
        if (!optimalProgramArr[truthTable].nOps ||
            optimalProgramArr[truthTable].nOps > base.program.nOps + nCurrentLevel) {
            if (!optimalProgramArr[truthTable].nOps) {
                ++nFilledTruthTables;
            }
            for (u32 i = 0; i < base.program.nOps - 1; ++i) {
                optimalProgramArr[truthTable].opArr[i] = base.program.opArr[i];
            }
            for (u32 i = 0; i <= nCurrentLevel + 1; ++i) {
                optimalProgramArr[truthTable].opArr[base.program.nOps - 1 + i] = frameArr[i].op;
            }
            optimalProgramArr[truthTable].nOps = base.program.nOps + nCurrentLevel;
        }
        if (!(nValidProgramsFound & 0xffffff)) {
            printStatus();
            printLastEvaluated(base, frameArr, nCurrentLevel, truthTable);
        }
    }
}

// GPU

void gpuSearch(const BaseVec& baseVec)
{
    cudaProfilerStart();

    u32 nThreads = static_cast<u32>(baseVec.size());
//    u32 nThreads = 32;
    int nThreadsPerBlock = 1024;
    int nBlocksPerGrid = (nThreads + nThreadsPerBlock - 1) / nThreadsPerBlock;

    Base* dBaseVec;
    cudaMallocManaged(&dBaseVec, nThreads * sizeof(Base));
    memcpy(dBaseVec, &baseVec[0], nThreads * sizeof(Base));
    cudaDeviceSynchronize();
    searchKernel<<<nBlocksPerGrid, nThreadsPerBlock>>>(nThreads, dBaseVec);
    cudaDeviceSynchronize();

    cudaError_t err(cudaGetLastError());
	if (err != cudaSuccess) {
		print("Error: Kernel launch failed: {}\n", cudaGetErrorString(err));
	}

    cudaProfilerStop();
    cudaDeviceReset();
}

__global__ void searchKernel(u32 nThreads, const Base* baseVec)
{
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nThreads) {
        return;
    }
    gpuCheckBase(baseVec[i]);
}

__device__ void gpuCheckBase(const Base& base)
{
    ::printf("%d nOps=%d\n", threadIdx.x, base.program.nOps);
    ::printf("%d nStackItems=%d\n", threadIdx.x, base.frame.nStackItems);
    for (int i = 0; i < base.frame.nStackItems; ++i) {
        ::printf(" %d", base.frame.stack[i]);
    }
    ::printf("\n");

    u32 nCurrentLevel = 0;
    u32 nSearchLevels = nSearchLevelsThread - nSearchLevelsBase + 1 /* nop */;
    Frame frameArr[nSearchLevelsThread]; // nSearchLevels
    memset(frameArr, 0, sizeof(frameArr)); ///////////////
    //memcpy(frameArr, &base.frame, sizeof(base.frame));
    frameArr[0] = base.frame;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1, nSearchLevels, false))) {
        Bits truthTable = frameArr[nCurrentLevel].stack[0];
//        // TODO: CUDA PROFILING
//        u64 old = atomicAdd(reinterpret_cast<unsigned long long*>(&nValidProgramsFound), 1);
//        if (old > 10000) {
//            return;
//        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&nFoundPrograms[truthTable]), 1);
    }
}

void printStatus()
{
    Fsec elapsedSec = Time::now() - startTime;
    print("\nWalltime: {} ({:.2f}s)\n", secondsToHms(elapsedSec.count()), elapsedSec.count());
    print("Filled truth tables: {} ({:.2f}%)\n", nFilledTruthTables, static_cast<float>(nFilledTruthTables) / nTotalTruthTables * 100.0f);
    print("Valid programs: {} ({:d} per sec)\n", nValidProgramsFound, static_cast<u32>(nValidProgramsFound / elapsedSec.count()));
}

void printLastEvaluated(const Base& base, const FrameArr& frameArr, u32 nCurrentLevel, Bits truthTable)
{
    print("Last evaluated: {:016b}:{} -{} ({} ops)\n", truthTable, serializeBase(base),
          serializeProgram(frameArr, nCurrentLevel), base.program.nOps + nCurrentLevel);
}

void testProgramGenerator()
{
    FrameArr frameArr;
    frameArr[0].op = 0; // nop
    frameArr[0].nStackItems = 0;
    evalOperation(frameArr[0]);
    u32 nCurrentLevel = 0;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1, nSearchLevelsThread, false))) {
        print("{:<50}{}\n", "## RECEIVED ##", serializeProgram(frameArr, nCurrentLevel));
    }
}

inline bool opIsLoad(const Base& base, u32 i) {
    return base.program.opArr[i] >= V1 && base.program.opArr[i] <= V4;
}

inline bool opIsAndOr(const Base& base, u32 i) {
    return base.program.opArr[i] == AND || base.program.opArr[i] == OR;
}

struct less_than_key
{
    inline bool operator() (const Base& a, const Base& b)
    {
//        if (b.program.nOps > a.program.nOps) {
////            print(">\n");
//            return false;
//        }
        for (int i = a.program.nOps - 1; i >= 0; --i) {
//            if (opIsLoad(a, i) && opIsLoad(b, i)) {
//                continue;
//            }
//            if (opIsAndOr(a, i) && opIsAndOr(b, i)) {
//                continue;
//            }
            if (a.program.opArr[i] < b.program.opArr[i]) {
                return true;
            }
            if (a.program.opArr[i] > b.program.opArr[i]) {
                return false;
            }
        }
        assert(false);
        return false;
    }
};

struct less_than_key2
{
    inline bool operator() (const Base& a, const Base& b)
    {
//        if (b.program.nOps > a.program.nOps) {
////            print(">\n");
//            return false;
//        }
        for (int i = a.program.nOps - 1; i >= 0; --i) {
//            if (opIsLoad(a, i) && opIsLoad(b, i)) {
//                continue;
//            }
//            if (opIsAndOr(a, i) && opIsAndOr(b, i)) {
//                continue;
//            }
            if (a.program.opArr[i] < b.program.opArr[i]) {
                return true;
            }
            if (a.program.opArr[i] > b.program.opArr[i]) {
                return false;
            }
        }
        assert(false);
        return false;
    }
};

struct less_than_key3
{
    inline bool operator() (const Base& a, const Base& b)
    {
        for (int i = a.program.nOps - 1; i >= 0; --i) {
            if (i == a.program.nOps - 3) {
                if (a.frame.nStackItems < b.frame.nStackItems) {
                    return true;
                }
                if (a.frame.nStackItems > b.frame.nStackItems) {
                    return false;
                }
            }
            if (a.program.opArr[i] < b.program.opArr[i]) {
                return true;
            }
            if (a.program.opArr[i] > b.program.opArr[i]) {
                return false;
            }
        }
        assert(false);
        return false;
    }
};

BaseVec generateBases()
{
    FrameArr frameArr;
    frameArr[0].op = NOP;
    frameArr[0].nStackItems = 0;
    u32 nCurrentLevel = 0;  // 1 is first real level, 0 is always the NOP base.
    BaseVec baseVec;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1, nSearchLevelsBase + 1 /* nop */, true))) {
//        print("{:<50}{}\n", "## RECEIVED ##", serializeProgram(frameArr, nCurrentLevel));
        if (nCurrentLevel != nSearchLevelsBase) {
            continue;
        }
//        print("{:<50}{}\n", "Base", serializeProgram(frameArr, nCurrentLevel));
        Base base;
        for (u32 i = 1; i <= nCurrentLevel; ++i) {
            base.program.opArr[i - 1] = frameArr[i].op;
        }
        base.program.nOps = nCurrentLevel;
        base.frame = frameArr[nCurrentLevel];
        baseVec.push_back(base);
    }
    sort(baseVec.begin(), baseVec.end(), less_than_key3());
    return baseVec;
}

void searchBase()
{
    FrameArr frameArr;
    frameArr[0].op = NOP;
    frameArr[0].nStackItems = 0;
    u32 nCurrentLevel = 0;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1, nSearchLevelsBase, false))) {
        Bits truthTable = frameArr[nCurrentLevel].stack[0];
        optimalProgramArr[truthTable].nOps = nCurrentLevel;
        for (u32 i = 1; i <= nCurrentLevel; ++i) { // skip nop
            optimalProgramArr[truthTable].opArr[i - 1] = frameArr[i].op;
        }
    }
}

__host__ __device__ u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels, u32 nSearchLevels, bool makeBases)
{
//    print("{:<50}{}\n", "Entering", serializeProgram(frameArr, nCurrentLevel));
    bool descendIfPossible = true;
    while (true) {
        bool newLevel = false;
        if (descendIfPossible) {
            descendIfPossible = false;
            // Skip if we're already at the lowest level.
            if (nCurrentLevel == nSearchLevels - 1) {
                continue;
            }
            // Enter branch.
            ++nCurrentLevel;
            newLevel = true;
        }
        if (newLevel) {
            frameArr[nCurrentLevel].op = 1;
//            print("{:<50}{}\n", "Descended to new level", serializeProgram(frameArr, nCurrentLevel));
        }
        else {
            frameArr[nCurrentLevel].op += 1;
            if (frameArr[nCurrentLevel].op == nOps) {
                if (--nCurrentLevel < nBaseLevels) {
                    return nCurrentLevel;
                }
//                print("{:<50}{}\n", "Ascended to higher level, checked earlier", serializeProgram(frameArr, nCurrentLevel));
                continue;
            }
        }
//        // FOR TESTING: Skip branch that has more than 3 values on the stack.
//        if (frameArr[nCurrentLevel - 1].nStackItems > 3) {
//            continue;
//        }
        // Skip branch that starts with (Vx Vy and/or) when x >= y
        if (nCurrentLevel >= 2
            && frameArr[nCurrentLevel].op >= AND && frameArr[nCurrentLevel].op <= OR
            && frameArr[nCurrentLevel - 2].op >= V1 && frameArr[nCurrentLevel - 1].op <= V4
            && frameArr[nCurrentLevel - 2].op >= frameArr[nCurrentLevel - 1].op) {
                continue;
        }
        // Skip branch that starts with (Vx Vy eor) when x > y
        if (nCurrentLevel >= 2
            && frameArr[nCurrentLevel].op == EOR
            && frameArr[nCurrentLevel - 2].op >= V1 && frameArr[nCurrentLevel - 1].op <= V4
            && frameArr[nCurrentLevel - 2].op > frameArr[nCurrentLevel - 1].op) {
                continue;
        }
        // Only enter and/or/eor branch when stackDepth > 1
        if (frameArr[nCurrentLevel - 1].nStackItems <= 1 && frameArr[nCurrentLevel].op >= AND && frameArr[nCurrentLevel].op <= EOR) {
            continue;
        }
        // Only enter load branch when stackDepth < remaining
        u32 nUnusedLevels = nSearchLevels - nCurrentLevel;
        if (frameArr[nCurrentLevel - 1].nStackItems > nUnusedLevels + 1) {
            continue;
        }
        // Only enter not branch when stackDepth <= remaining && stackDepth > 0
        if (frameArr[nCurrentLevel].op == NOT && ! (frameArr[nCurrentLevel - 1].nStackItems <= nUnusedLevels && frameArr[nCurrentLevel - 1].nStackItems > 0)) {
            continue;
        }
        // Don't enter branch which ends in (not not)
        if (frameArr[nCurrentLevel - 1].op == NOT && frameArr[nCurrentLevel].op == NOT) {
            continue;
        }
        // Copy the stack from the previous level.
        frameArr[nCurrentLevel].nStackItems = frameArr[nCurrentLevel - 1].nStackItems;
        for (u32 i = 0; i < frameArr[nCurrentLevel].nStackItems; ++i) {
            frameArr[nCurrentLevel].stack[i] = frameArr[nCurrentLevel - 1].stack[i];
        }
        // Evaluate the next operation on top of the new stack.
        evalOperation(frameArr[nCurrentLevel]);

        if (makeBases) {
            return nCurrentLevel;
        }

        // This optimization can't be used in current CUDA version. It causes the threads to fall out of sync.
//        // Don't enter branch which has a base in which a shorter program already exists in the table of optimal programs
//        if (frameArr[nCurrentLevel].nStackItems == 1) {
//            Bits truthTable = frameArr[nCurrentLevel].stack[0];
//            if (optimalProgramArr[truthTable].nOps && optimalProgramArr[truthTable].nOps <= nCurrentLevel + 1) {
//                continue;
//            }
//        }
        if (frameArr[nCurrentLevel].nStackItems != 1) {
            descendIfPossible = true;
            continue;
        }
//      print("{:<50}{} results={}\n", "Skipping program that returns <> 1 results", serializeProgram(frameArr, nCurrentLevel), frameArr[nCurrentLevel].nStackItems);
//      print("{:<50}{}\n", "Returning valid program", serializeProgram(frameArr, nCurrentLevel));
        return nCurrentLevel;
    }
}

__host__ __device__ void evalOperation(Frame& f)
{
    switch (f.op) {
        case NOP: assert(false);
        case V1: push(f, 0b1010101010101010); break;
        case V2: push(f, 0b1100110011001100); break;
        case V3: push(f, 0b1111000011110000); break;
        case V4: push(f, 0b1111111100000000); break;
        case AND: push(f, pop(f) & pop(f)); break;
        case OR: push(f, pop(f) | pop(f)); break;
        case EOR: push(f, pop(f) ^ pop(f)); break;
        case NOT: push(f, ~pop(f)); break;
        default: assert(false);
    }
}

__host__ __device__ void push(Frame& f, Bits v)
{
    f.stack[f.nStackItems] = v;
    ++f.nStackItems;
}

__host__ __device__ Bits pop(Frame& f)
{
    --f.nStackItems;
    return f.stack[f.nStackItems];
}

void writeResults(const string& path, const ProgramArr& programArr)
{
    ofstream f(path, ios::out);
    for (u32 i = 0; i < nTotalTruthTables; ++i) {
        f << format("{:016b}:", i);
        if (programArr[i].nOps) {
            for (u32 j = 0; j < programArr[i].nOps; ++j) {
                f << format(" {}", opStrArr[programArr[i].opArr[j]]);
            }
            f << format(" ({} ops)", programArr[i].nOps);
        }
        else {
            f << " <none>";
        }
        f << format("\n");
    }
}

void writeHistogram(const string& path)
{
    ofstream f(path, ios::out);
    for (u32 i = 0; i < nTotalTruthTables; ++i) {
        f << format("{:016b}: {}\n", i, nFoundPrograms[i]);
    }
}

string serializeProgram(const FrameArr& frameArr, u32 nFrames)
{
    stringstream ss;
    for (u32 i = 1; i <= nFrames; ++i) { // skip the first op, which is nop or base connection
        ss << format(" {}", opStrArr[frameArr[i].op]);
    }
    return ss.str();
}

string serializeBase(const Base& base)
{
    stringstream ss;
    for (u32 i = 0; i < base.program.nOps; ++i) {
        ss << format(" {}", opStrArr[base.program.opArr[i]]);
    }
    ss << format(" (s={})", base.frame.nStackItems);
    return ss.str();
}

string serializeFrame(const Frame& f)
{
    stringstream ss;
    ss << format("Frame: op={} nStack={} stack=", opStrArr[f.op], f.nStackItems);
    for (u32 i = 0; i < f.nStackItems; ++i) {
        ss << format("{:016b}", f.stack[i]);
    }
    return ss.str();
}

string secondsToHms(double sec)
{
    u64 n = static_cast<u64>(sec);
    u64 s = n % 60;
    n /= 60;
    u64 m = n % 60;
    n /= 60;
    u64 h = n % 60;
    return format("{:02d}:{:02d}:{:02d}", h, m, s);
}
