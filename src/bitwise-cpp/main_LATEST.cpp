// Style: http://geosoft.no/development/cppstyle.html

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>

#include <cppformat/format.h>

#include "int_types.h"

using namespace std;
using namespace fmt;

// 8 operations
const u32 nOps = 8;
const u32 opStackRequireArr[] = {0, 0, 0, 0, 2, 2, 2, 1};
const char* opStrArr[] = {"v1", "v2", "v3", "v4", "and", "or", "eor", "not"};

// Each op may leave the stack unchanged or add or remove one value. Since the program must leave exactly one value on
// the stack, the program cannot possibly recover if, at any given point while running the program, nStackItems + 1 >
// nRemainingSearchLevels. So, if programs are stopped when they become unable to recover, the maximum possible stack
// usage occurs when each op adds one value until the program is stopped.
const u32 nSearchLevels = 5;
const u32 maxStackItems = nSearchLevels / 2 + 1;
const u32 nTotalTruthTables = 1 << 16;

typedef u16 Bits;
typedef Bits BitsStack[maxStackItems];
struct Frame {
    u32 op;
    BitsStack stack;
    u32 nStackItems;
};
typedef Frame FrameArr[nSearchLevels];
struct Program {
    u32 opArr[nSearchLevels];
    u32 nOps;
};
typedef Program ProgramArr[nTotalTruthTables];

ProgramArr optimalProgramArr;

typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

inline void testProgramGenerator();
inline u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels);
inline Frame nextFrame(Frame frame, u32 op);
inline void evalOperation(Frame&);
inline void push(Frame& f, Bits v);
inline Bits pop(Frame& s);
void writeResults(const string& resultPath, const ProgramArr& optimalProgramArr);
string serializeProgram(const FrameArr&, u32 nFrames);
string serializeFrame(const Frame& f);

int main()
{
	// Switch from C locale to user's locale. This will typically cause integers to be printed with thousands
	// separators.
	locale::global(locale(""));
	cout.imbue(locale(""));

    print("Search Levels: {}\n", nSearchLevels);
    print("Maximum Stack Items: {}\n", maxStackItems);

//    testProgramGenerator();
//    return 0;

    //ProgramArr optimalProgramArr;
    memset(optimalProgramArr, 0, sizeof(optimalProgramArr));

    u32 nFilledTruthTables = 0;
    #pragma omp parallel for
    // Start 4 threads with bases v1 to v4.
    for (u32 op = 0; op <= 3; ++op) {
        FrameArr frameArr;
        frameArr[0].op = op;
        frameArr[0].nStackItems = 0;
        evalOperation(frameArr[0]);

        u32 nCurrentLevel = 0;
        auto startTime = Time::now();
        #pragma omp critical
        {
            print("Starting thread {}/{} with program: {}\n", omp_get_thread_num(), omp_get_num_threads(),
                  serializeProgram(frameArr, 1));
        }
        u64 nValidProgramsFound = 0;

        while (true) {
            ++nValidProgramsFound;
            Bits truthTable = frameArr[nCurrentLevel].stack[0];
            if (!optimalProgramArr[truthTable].nOps || optimalProgramArr[truthTable].nOps > nCurrentLevel) {
                #pragma omp critical
                {
                    if (!optimalProgramArr[truthTable].nOps) {
                        ++nFilledTruthTables;
                    }
//                    print("{} {:016b}: {} ({} ops)\n", opIdx, truthTable, serializeProgram(frameArr, nCurrentLevel + 1), nCurrentLevel + 1);
                    optimalProgramArr[truthTable].nOps = nCurrentLevel + 1;
                    for (u32 i = 0; i < nCurrentLevel; ++i) {
                        optimalProgramArr[truthTable].opArr[i] = frameArr[i].op;
                    }
                }
            }
            if (!op && ((!(nValidProgramsFound & 0xffffff)) || nFilledTruthTables == nTotalTruthTables)) {
                #pragma omp critical
                {
                    Fsec elapsedSec = Time::now() - startTime;
                    print("\nThread: {}\n", op);
                    print("Walltime, this thread: {:.2f}s\n", elapsedSec.count());
                    print("Valid found, this thread: {} ({:d} per sec)\n", nValidProgramsFound,
                          static_cast<u32>(nValidProgramsFound / elapsedSec.count()));
                    print("Filled truth tables, all threads: {} ({:.2f}%)\n", nFilledTruthTables,
                          static_cast<float>(nFilledTruthTables) / nTotalTruthTables * 100.0f);
                    print("Last evaluated: {} ({} ops)\n", serializeProgram(frameArr, nCurrentLevel + 1), nCurrentLevel + 1);
                }
            }
            if (!(nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1))) {
                break;
            }
        }
    }
    writeResults("bitwise.txt", optimalProgramArr);
//    getchar();
    return 0;
}

void testProgramGenerator()
{
    FrameArr frameArr;
    frameArr[0].op = 1; // v2
    frameArr[0].nStackItems = 0;
    evalOperation(frameArr[0]);
    u32 nCurrentLevel = 0;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1))) {
        print("{:<50}{}\n", "######## RECEIVED ########", serializeProgram(frameArr, nCurrentLevel + 1));
    }
}

// TODO: Support nBaseLevels = 0
// Given a current program, generate the next valid program. Return false when there are no more programs.
u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels)
{
//    print("{:<50}{}\n", "Entering", serializeProgram(frameArr, nCurrentLevel + 1));
    bool descendIfPossible = true;
    while (true) {
        bool newLevel = false;
        if (descendIfPossible) {
            descendIfPossible = false;
            if (nCurrentLevel < nSearchLevels - 1) {
                u32 nUnusedLevels = nSearchLevels - nCurrentLevel;
                if (frameArr[nCurrentLevel].nStackItems <= nUnusedLevels + 1) {
                    ++nCurrentLevel;
                    newLevel = true;
                }
                else {
//                    print("{:<50}{} stackSize={} nUnusedLevels={}\n", "Skipped hopeless branch",
//                          serializeProgram(frameArr, nCurrentLevel + 1), frameArr[nCurrentLevel].nStackItems, nUnusedLevels);
                }
            }
        }
        if (newLevel) {
            frameArr[nCurrentLevel].op = 0;
//            print("{:<50}{}\n", "Descended to new level", serializeProgram(frameArr, nCurrentLevel + 1));
        }
        else {
            frameArr[nCurrentLevel].op += 1;
            if (frameArr[nCurrentLevel].op == nOps) {
                --nCurrentLevel;
                if (nCurrentLevel < nBaseLevels) {
                    return nCurrentLevel;
                }
//                print("{:<50}{}\n", "Ascended to higher level, checked earlier", serializeProgram(frameArr, nCurrentLevel + 1));
                continue;
            }
        }
        if (frameArr[nCurrentLevel - 1].nStackItems < opStackRequireArr[frameArr[nCurrentLevel].op]) {
//            print("{:<50}{}\n", "Skipping program that underflows", serializeProgram(frameArr, nCurrentLevel + 1));
            continue;
        }
        for (u32 i = 0; i < frameArr[nCurrentLevel - 1].nStackItems; ++i) {
            frameArr[nCurrentLevel].stack[i] = frameArr[nCurrentLevel - 1].stack[i];
        }
        frameArr[nCurrentLevel].nStackItems = frameArr[nCurrentLevel - 1].nStackItems;
        evalOperation(frameArr[nCurrentLevel]);
        if (frameArr[nCurrentLevel].nStackItems != 1) {
//            print("{:<50}{} results={}\n", "Skipping program that returns <> 1 results", serializeProgram(frameArr, nCurrentLevel + 1), frameArr[nCurrentLevel].nStackItems);
            descendIfPossible = true;
            continue;
        }
//        print("{:<50}{}\n", "Returning valid program", serializeProgram(frameArr, nCurrentLevel + 1));
        return nCurrentLevel;
    }
}

void evalOperation(Frame& f)
{
    switch (f.op) {
        case 0: push(f, 0b0000000011111111); break; // v1
        case 1: push(f, 0b0000111100001111); break; // v2
        case 2: push(f, 0b0011001100110011); break; // v3
        case 3: push(f, 0b0101010101010101); break; // v4
        case 4: push(f, pop(f) & pop(f)); break; // and
        case 5: push(f, pop(f) | pop(f)); break; // or
        case 6: push(f, pop(f) ^ pop(f)); break; // eor
        case 7: push(f, ~pop(f)); break; // not
        default: assert(false);
    }
}

void push(Frame& f, Bits v)
{
    f.stack[f.nStackItems] = v;
    ++f.nStackItems;
}

Bits pop(Frame& f)
{
    --f.nStackItems;
    return f.stack[f.nStackItems];
}

void writeResults(const string& resultPath, const ProgramArr& programArr)
{
    ofstream f(resultPath, ios::out);
    for (u32 i = 0; i < nTotalTruthTables; ++i) {
        f << format("{:016b}:", i);
        for (u32 j = 0; j < programArr[i].nOps; ++j) {
            f << format(" {}", opStrArr[programArr[i].opArr[j]]);
        }
        f << format("\n");
    }
}

string serializeProgram(const FrameArr& frameArr, u32 nFrames)
{
    stringstream ss;
    for (u32 i = 0; i < nFrames; ++i) {
        ss << format("{} ", opStrArr[frameArr[i].op]);
    }
    return ss.str();
}

string serializeFrame(const Frame& f)
{
    stringstream ss;
    ss << format("Frame: op={} nStack={} stack=", opStrArr[f.op], f.nStackItems);
    for (u32 i = 0; i < f.nStackItems; ++i) {
        ss << format("{:016b} ", f.stack[i]);
    }
    return ss.str();
}
