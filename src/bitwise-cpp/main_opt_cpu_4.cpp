// Style: http://geosoft.no/development/cppstyle.html

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>

#include <cppformat/format.h>

#include "int_types.h"

using namespace std;
using namespace fmt;

// 8 operations
const u32 nOps = 8;
const u32 opStackRequireArr[] = {0, 0, 0, 0, 2, 2, 2, 1};
const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "and", "or", "eor", "not"};

typedef u16 Bits;
typedef vector<Bits> BitsStack;
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;
struct Frame {
    u32 op;
    BitsStack stack;
};
typedef vector<Frame> FrameVec;
typedef vector<FrameVec> OptimalProgramVec;

inline void testProgramGenerator(u32 nSearchLevels);
inline u32 nextValidProgram(FrameVec& frameVec, u32 nCurrentLevel, u32 nSearchLevels, u32 nBaseLevels);
inline Frame nextFrame(Frame frame, u32 op);
inline void evalOperation(BitsStack& s, u32 op);
inline Bits pop(BitsStack& s);
void writeResults(const string& resultPath, const OptimalProgramVec& programVec);
string serializeProgram(const FrameVec& frameVec, u32 nFrames);


int main()
{
    const u32 nSearchLevels = 5;
    const u32 nTotalTruthTables = 1 << 16;

	// Switch from C locale to user's locale. This will typically cause integers to be printed with thousands
	// separators.
	locale::global(locale(""));
	cout.imbue(locale(""));

    testProgramGenerator(10);
    return 0;

    OptimalProgramVec optimalProgramVec(nTotalTruthTables);
    u32 nFilledTruthTables = 0;
    #pragma omp parallel for
    // Start 4 threads with bases v1 to v4.
    for (u32 op = 0; op <= 3; ++op) {
        FrameVec frameVec(nSearchLevels);
        frameVec[0].op = op;
        u32 nCurrentlevel = 0;
        evalOperation(frameVec[0].stack, op);
        auto startTime = Time::now();
        #pragma omp critical
        {
            print("Starting thread {}/{} with program: {}\n", omp_get_thread_num(), omp_get_num_threads(),
                  serializeProgram(frameVec, 1));
        }
        u64 nValidProgramsFound = 0;

        while (true) {
            ++nValidProgramsFound;
            Bits truthTable = frameVec[nCurrentlevel].stack.back();
            if (!optimalProgramVec[truthTable].size() || optimalProgramVec[truthTable].size() > nCurrentlevel) {
                #pragma omp critical
                {
                    if (!optimalProgramVec[truthTable].size()) {
                        ++nFilledTruthTables;
                    }
//                    print("{} {:016b}: {} ({} ops)\n", opIdx, truthTable, serializeProgram(frameVec, nCurrentlevel + 1), nCurrentlevel + 1);
                    optimalProgramVec[truthTable] = FrameVec(frameVec.begin(), frameVec.begin() + nCurrentlevel + 1);
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
                    print("Last evaluated: {} ({} ops)\n", serializeProgram(frameVec, nCurrentlevel + 1), nCurrentlevel + 1);
                }
            }
            if (!(nCurrentlevel = nextValidProgram(frameVec, nCurrentlevel, 1, nSearchLevels))) {
                break;
            }
        }
    }
    writeResults("bitwise.txt", optimalProgramVec);
//    getchar();
    return 0;
}

u64 t = 0;

void testProgramGenerator(u32 nSearchLevels)
{
    FrameVec frameVec(nSearchLevels);
    // We assume that all programs in the base are checked.
    frameVec[0].op = 1; // v2
    evalOperation(frameVec[0].stack, frameVec[0].op);
    u32 nCurrentLevel = 0;
    while ((nCurrentLevel = nextValidProgram(frameVec, nCurrentLevel, 1, nSearchLevels))) {
//        print("{:<50}{}\n", "########## RECEIVED ##########", serializeProgram(frameVec, nCurrentLevel + 1));
    }
    print("{}\n", t);
}

// TODO: Support nBaseLevels = 0
// Given a current program, generate the next valid program. Return false when there are no more programs.
u32 nextValidProgram(FrameVec& frameVec, u32 nCurrentLevel, u32 nBaseLevels, u32 nSearchLevels)
{
    // When the function is called, we start by checking if it's possible to descend one step down. If that's possible,
    // we descend and return the program if it's valid.
    bool descendIfPossible = true;
    while (true) {
        bool newLevel = false;
        if (descendIfPossible) {
            descendIfPossible = false;
            // Only descend if we're not already at the lowest level.
            if (nCurrentLevel < nSearchLevels - 1) {
                // Skip branches that cannot possibly recover because there are more values on the stack than there are
                // remaining levels (each op can remove at most one value from the stack).
                if (frameVec[nCurrentLevel].stack.size() <= static_cast<u32>(nSearchLevels - nCurrentLevel + 1)) {
                    ++nCurrentLevel;
                    newLevel = true;
                }
                else {
//                    print("{:<50}{} stackSize={} nUnusedLevels={}\n", "Skipped hopeless branch",
//                          serializeProgram(frameVec, nCurrentLevel + 1), frameVec[nCurrentLevel].stack.size(), nUnusedLevels);
                }
            }
        }
        u32 nextOp;
        if (newLevel) {
            frameVec[nCurrentLevel].op = 0;
            nextOp = 0;
        }
        else {
            nextOp = frameVec[nCurrentLevel].op + 1;
            frameVec[nCurrentLevel].op = nextOp;
            if (nextOp == nOps) {
                --nCurrentLevel;
                if (nCurrentLevel < nBaseLevels) {
                    // Back at the base program. Iteration is complete after returning this as the last valid program.
                    return nCurrentLevel;
                }
//                print("{:<50}{}\n", "Ascend to higher level, checked earlier", serializeProgram(frameVec, nCurrentLevel + 1));
                // We're back to a level that has already been returned. We jump to the top to generate the next
                // variation on the same level. If the new variation passes the tests, it is returned as the next valid
                // program and then becomes the root for further programs.
                continue;
            }
        }
        ++t;
        if (frameVec[nCurrentLevel - 1].stack.size() < opStackRequireArr[nextOp]) {
//            print("{:<50}{}\n", "Skipping program that underflows", serializeProgram(frameVec, nCurrentLevel + 1));
            // Generate the next variation while staying on the same level. This causes a move away from the current
            // program, which prevents it from becoming the root for any new programs, all of which would be invalid.
            continue;
        }
        frameVec[nCurrentLevel].stack = frameVec[nCurrentLevel - 1].stack;
        evalOperation(frameVec[nCurrentLevel].stack, nextOp);
        if (frameVec[nCurrentLevel].stack.size() != 1) {
//            print("{:<50}{} results={}\n", "Skipping program that returns <> 1 results", serializeProgram(frameVec, nCurrentLevel + 1), frameVec[nCurrentLevel].stack.size());
            // Move to a lower level if possible. Programs that don't underflow but return a stack with more or less
            // than one item may become valid when adding one or more ops, so those branches must be searched.
            descendIfPossible = true;
            continue;
        }
//        print("{:<50}{}\n", "Returning valid program", serializeProgram(frameVec, nCurrentLevel + 1));
        return nCurrentLevel;
    }
}

void evalOperation(BitsStack& s, u32 op)
{
    switch (op) {
        case 0: s.push_back(0b0000000011111111); // v1
            break;
        case 1: s.push_back(0b0000111100001111); // v2
            break;
        case 2: s.push_back(0b0011001100110011); // v3
            break;
        case 3: s.push_back(0b0101010101010101); // v4
            break;
        case 4: s.push_back(pop(s) & pop(s)); // and
            break;
        case 5: s.push_back(pop(s) | pop(s)); // or
            break;
        case 6: s.push_back(pop(s) ^ pop(s)); // eor
            break;
        case 7: s.push_back(~pop(s)); // not
            break;
        default: assert(false);
            break;
    }
}

Bits pop(BitsStack& s)
{
    Bits v = s.back();
    s.pop_back();
    return v;
}

void writeResults(const string& resultPath, const OptimalProgramVec& programVec)
{
    ofstream f(resultPath, ios::out);
    Bits truthTable = 0;
    for (auto program : programVec) {
        if (program.size()) {
            f << format("{:016b}: {}\n", truthTable, serializeProgram(program, static_cast<u32>(program.size())));
        }
        ++truthTable;
    }
}

string serializeProgram(const FrameVec& frameVec, u32 nFrames)
{
    string s;
    for (u32 i = 0; i < nFrames; ++i) {
        s += format("{} ", opStrVec[frameVec[i].op]);
    }
    return s;
}
