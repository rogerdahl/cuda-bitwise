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
const u32 nOps = 9;
const char* opStrArr[] = {"nop", "v1", "v2", "v3", "v4", "and", "or", "eor", "not"};
const u32 NOP = 0, V1 = 1, V2 = 2, V3 = 3, V4 = 4, AND = 5, OR = 6, EOR = 7, NOT = 8;
// Each op may leave the stack unchanged or add or remove one value. Since the program must leave exactly one value on
// the stack, the program cannot possibly recover if, at any given point while running the program, nStackItems + 1 >
// nRemainingSearchLevels. So, if programs are stopped when they become unable to recover, the maximum possible stack
// usage occurs when each op adds one value until the program is stopped.
const u32 nSearchLevels = 18;
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
u32 nFilledTruthTables = 0;
u64 nValidProgramsFound = 0;

typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

void printStatus(const FrameArr& frameArr, u32 nCurrentLevel);
inline void testProgramGenerator();
inline u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels);
inline void evalOperation(Frame&);
inline void push(Frame& f, Bits v);
inline Bits pop(Frame& s);
void writeResults(const string& resultPath, const ProgramArr& optimalProgramArr);
string serializeProgram(const FrameArr&, u32 nFrames);
string serializeFrame(const Frame& f);
string secondsToHms(double sec);

auto startTime = Time::now();

int main()
{
	// Switch from C locale to user's locale. This will typically cause integers to be printed with thousands
	// separators.
	locale::global(locale(""));
	cout.imbue(locale(""));

    print("Search levels: {}\n", nSearchLevels);
    print("Max stack items: {}\n", maxStackItems);

//    testProgramGenerator();
//    return 0;

    //ProgramArr optimalProgramArr;
    memset(optimalProgramArr, 0, sizeof(optimalProgramArr));

    #pragma omp parallel for
    // Start 4 threads with bases v1 to v4.
    for (u32 op = V1; op <= V4; ++op) {
        Frame frameArr[nSearchLevels];
        frameArr[0].op = op;
        frameArr[0].nStackItems = 0;
        evalOperation(frameArr[0]);
        u32 nCurrentLevel = 0;
        #pragma omp critical
        {
            print("Starting thread {}/{} with program: {}\n", omp_get_thread_num(), omp_get_num_threads(), serializeProgram(frameArr, 0));
        }
        do {
            #pragma omp atomic
            ++nValidProgramsFound;

            Bits truthTable = frameArr[nCurrentLevel].stack[0];
            if (!optimalProgramArr[truthTable].nOps || optimalProgramArr[truthTable].nOps > nCurrentLevel) {
                #pragma omp critical
                {
                    if (!optimalProgramArr[truthTable].nOps) {
                        ++nFilledTruthTables;
                    }
//                    print("{} {:016b}: {} ({} ops)\n", opIdx, truthTable, serializeProgram(frameArr, nCurrentLevel), nCurrentLevel + 1);
                    optimalProgramArr[truthTable].nOps = nCurrentLevel + 1;
                    for (u32 i = 0; i <= nCurrentLevel; ++i) {
                        optimalProgramArr[truthTable].opArr[i] = frameArr[i].op;
                    }
                }
            }
            if (op == V1 && (!(nValidProgramsFound & 0xff))) {
//            if (op == V1) {
                printStatus(frameArr, nCurrentLevel);
            }
        } while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1)));

        if (op == V1) {
            print("Final:\n");
            printStatus(frameArr, nCurrentLevel);
        }
    }
    writeResults("bitwise.txt", optimalProgramArr);
    return 0;
}

void printStatus(const FrameArr& frameArr, u32 nCurrentLevel)
{
    Fsec elapsedSec = Time::now() - startTime;
    print("\nWalltime: {} ({:.2f}s)\n", secondsToHms(elapsedSec.count()), elapsedSec.count());
    print("Filled truth tables: {} ({:.2f}%)\n", nFilledTruthTables, static_cast<float>(nFilledTruthTables) / nTotalTruthTables * 100.0f);
    print("Valid programs: {} ({:d} per sec)\n", nValidProgramsFound, static_cast<u32>(nValidProgramsFound / elapsedSec.count()));
    print("Last evaluated, thread: {} ({} ops)\n", serializeProgram(frameArr, nCurrentLevel), nCurrentLevel + 1);
}

void testProgramGenerator()
{
    FrameArr frameArr;
    frameArr[0].op = 0; // nop
    frameArr[0].nStackItems = 0;
    evalOperation(frameArr[0]);
    u32 nCurrentLevel = 0;
    while ((nCurrentLevel = nextValidProgram(frameArr, nCurrentLevel, 1))) {
        print("{:<50}{}\n", "######## RECEIVED ########", serializeProgram(frameArr, nCurrentLevel));
    }
}

// Given a current program, generate the next valid program. Return false when there are no more programs.

// - only enter and/or/eor branch when: stackDepth > 1
// - only enter not branch when: stackDepth <= remaining && stackDepth > 0
// - only enter load branch when: stackDepth < remaining
// - don't enter branch which has a base in which a shorter program already exists in the table of optimal programs
// - don't enter branch which ends in (Vx Vy {and/or/eor}) when x >= y
// - don't enter branch which ends in (not not)
// - No need to check for underflow.

u32 nextValidProgram(FrameArr& frameArr, u32 nCurrentLevel, u32 nBaseLevels)
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
//        // FOR TESTING. MAY NOT FIND SHORTEST: Skip branch that has more than 3 values on the stack.
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
        // - don't enter branch which has a base in which a shorter program already exists in the table of optimal programs
        if (frameArr[nCurrentLevel].nStackItems == 1) {
            Bits truthTable = frameArr[nCurrentLevel].stack[0];
            if (optimalProgramArr[truthTable].nOps && optimalProgramArr[truthTable].nOps <= nCurrentLevel + 1) {
                continue;
            }
        }
        else {
            descendIfPossible = true;
            continue;
        }
//      print("{:<50}{} results={}\n", "Skipping program that returns <> 1 results", serializeProgram(frameArr, nCurrentLevel), frameArr[nCurrentLevel].nStackItems);
//      print("{:<50}{}\n", "Returning valid program", serializeProgram(frameArr, nCurrentLevel));
        return nCurrentLevel;
    }
}

void evalOperation(Frame& f)
{
    switch (f.op) {
        case 0: break; // nop
        case 1: push(f, 0b1010101010101010); break; // v1
        case 2: push(f, 0b1100110011001100); break; // v2
        case 3: push(f, 0b1111000011110000); break; // v3
        case 4: push(f, 0b1111111100000000); break; // v4
        case 5: push(f, pop(f) & pop(f)); break; // and
        case 6: push(f, pop(f) | pop(f)); break; // or
        case 7: push(f, pop(f) ^ pop(f)); break; // eor
        case 8: push(f, ~pop(f)); break; // not
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
        if (!programArr[i].nOps) {
            f << " <none>";
        }
        else {
            for (u32 j = 0; j < programArr[i].nOps; ++j) {
                f << format(" {}", opStrArr[programArr[i].opArr[j]]);
            }
        }
        f << format("\n");
    }
}

string serializeProgram(const FrameArr& frameArr, u32 nFrames)
{
    stringstream ss;
    for (u32 i = 0; i <= nFrames; ++i) {
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

