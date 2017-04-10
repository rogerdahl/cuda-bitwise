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
enum Operation  { LD_A, LD_B, LD_C, LD_D, AND, OR, XOR, NOT, ENUM_END };
//const s32 operationStackDiffArr[]{1, 1, 1, 1, -1, -1, -1, 0};
const u32 operationStackRequireArr[]{0, 0, 0, 0, 2, 2, 2, 1};
const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "and", "or", "xor", "not", "ENUM_END"};
// 12 operations
//enum Operation {
//    LD_A, LD_B, LD_C, LD_D, LD_ONES, LD_ZEROS, AND, OR, XOR, NOT, DUP, POP, ENUM_END
//};
//const s32 operationStackDiffArr[]{1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 1, -1};
//const s32 operationStackRequireArr[]{0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1};
//const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "ones", "zeros", "and", "or", "xor", "not", "dup", "pop", "ENUM_END"};

//typedef vector<Operation> ProgramVec;
typedef u16 Bits;
typedef vector<Bits> BitsStack;
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

struct Frame {
    Operation op;
    BitsStack stack;
};

typedef vector<Frame> FrameVec;
typedef vector<FrameVec> OptimalProgramVec;


//u64 countValidPrograms(u32 nSearchLevels);
void testProgramGenerator(u32 nSearchLevels);
inline bool nextValidProgram(FrameVec& frameVec, u32 nSearchLevels, u32 nBaseLevels);
inline Frame nextFrame(Frame frame, Operation operation);
//inline bool stackUnderflows(bool& hasOneStackItem, const ProgramVec& program);
//Bits evalProgram(const ProgramVec& program);
void evalOperation(BitsStack& s, const Operation operation);
inline Bits pop(BitsStack& s);
void writeResults(const string& resultPath, const OptimalProgramVec& programVec);
string serializeProgram(const FrameVec& frameVec);
u64 pow(u32 base, u32 exp);


int main()
{
    const u32 nSearchLevels = 15;
    const u32 nTotalTruthTables = 1 << 16;
    const u64 nTotalPossiblePrograms = pow(static_cast<int>(ENUM_END), nSearchLevels);

	// Switch from C locale to user's locale. This will typically cause integers to be printed with thousands
	// separators.
	locale::global(locale(""));
	cout.imbue(locale(""));

//    testProgramGenerator(4);
//    return 0;

    print("Total possible programs: {}\n", nTotalPossiblePrograms);
//    print("Total valid programs: {}\n", countValidPrograms(nSearchLevels));

    OptimalProgramVec optimalProgramVec(nTotalTruthTables);
    u32 nFilledTruthTables = 0;
    #pragma omp parallel for
    for (auto opIdx = static_cast<int>(LD_A); opIdx <= static_cast<int>(LD_D); opIdx++) {
        auto baseOperation = static_cast<Operation>(opIdx);
        auto baseFrame = nextFrame({ENUM_END, {}}, baseOperation);
        FrameVec frameVec = {baseFrame};
        auto startTime = Time::now();
        #pragma omp critical
        {
            print("Starting thread {}/{} with program: {}\n", omp_get_thread_num(), omp_get_num_threads(),
                  serializeProgram(frameVec));
        }
        u64 nValidProgramsFound = 0;
        while (true) {
            ++nValidProgramsFound;
            Bits truthTable = frameVec.back().stack.back();
            if (!optimalProgramVec[truthTable].size() || optimalProgramVec[truthTable].size() > frameVec.size()) {
                #pragma omp critical
                {
                    if (!optimalProgramVec[truthTable].size()) {
                        ++nFilledTruthTables;
                    }
//                    print("{} {:016b}: {}\n", opIdx, truthTable, serializeProgram(frameVec));
                    optimalProgramVec[truthTable] = frameVec;
                }
            }
            if (!opIdx && ((!(nValidProgramsFound & 0xfffff)) || nFilledTruthTables == nTotalTruthTables)) {
                #pragma omp critical
                {
                    Fsec elapsedSec = Time::now() - startTime;
                    print("\nThread: {}\n", opIdx);
                    print("Walltime, this thread: {:.2f}s\n", elapsedSec.count());
                    print("Valid found, this thread: {} ({:d} per sec)\n", nValidProgramsFound,
                          static_cast<u32>(nValidProgramsFound / elapsedSec.count()));
                    print("Filled truth tables, all threads: {} ({:.2f}%)\n", nFilledTruthTables,
                          static_cast<float>(nFilledTruthTables) / nTotalTruthTables * 100.0f);
                    print("Last evaluated: {} ({} ops)\n", serializeProgram(frameVec), frameVec.size());
                }
            }
            if (!nextValidProgram(frameVec, nSearchLevels, 1)) {
                break;
            }
        }
    }
    writeResults("bitwise.txt", optimalProgramVec);
//    getchar();
    return 0;
}

//u64 countValidPrograms(u32 nSearchLevels)
//{
//    u64 nValidPrograms = 1;
//    ProgramVec program = { LD_A };
//    while (nextValidProgram(program, nSearchLevels, 1)) {
//        ++nValidPrograms;
//        if (!(nValidPrograms & 0xffffff)) {
//            print("{}...\n", nValidPrograms);
//        }
//    }
//    return nValidPrograms;
//}

void testProgramGenerator(u32 nSearchLevels)
{
    auto baseFrame = nextFrame({ENUM_END, {}}, LD_A);
    FrameVec frameVec = {baseFrame};
    while (nextValidProgram(frameVec, nSearchLevels, 1));
}

// Given a current program, generate the next valid program. Return false when there are no more programs.
bool nextValidProgram(FrameVec& frameVec, u32 nSearchLevels, u32 nBaseLevels)
{
    bool descendIfPossible = true;
    while (true) {
        bool newLevel = false;
        // When the function is called, we start by checking if it's possible to descend one step down.
        // If that's possible, we descend and return the program if it's valid.
        if (descendIfPossible) {
            descendIfPossible = false;
            if (frameVec.size() < nSearchLevels) {
                // It's possible to descend...
                u32 nUnusedLevels = (u32) (nSearchLevels - frameVec.size());
                if (frameVec.back().stack.size() <= nUnusedLevels) {
                    // And also useful to descend. So do it.
                    frameVec.push_back(nextFrame(frameVec.back(), LD_A));
                    newLevel = true;
                }
                else {
//                    print("{:<50}{} stackSize={} nUnusedLevels={}\n", "Skipped hopeless branch.",
//                          serializeProgram(frameVec), frameVec.back().stack.size(), nUnusedLevels);
                }
            }
        }
        // "else" because we don't want to go to the next operation if pushed a new frame with LD_A to the stack.
        if (!newLevel) {
            auto newOpIdx = static_cast<int>(frameVec.back().op) + 1;
            frameVec.back().op = static_cast<Operation>(newOpIdx);
            if (frameVec.back().op == ENUM_END) {
                // We have iterated over all possible variations at this level. Drop back to a higher level.
                frameVec.pop_back();
                if (frameVec.size() == nBaseLevels) {
//                    print("{:<50}{}\n", "Back at the base program. Iteration is complete", serializeProgram(frameVec));
                    return false;
                }
//                print("{:<50}{}\n", "Ascend to higher level, checked earlier", serializeProgram(frameVec));
                // We're back to a level that has already been returned. We jump to the top to generate the next
                // variation on the same level. If the new variation passes the tests, it is returned as the next
                // valid program and then becomes the root for further programs.
                continue;
            }
        }
        auto opIdx = static_cast<int>(frameVec.back().op);
        if (frameVec.rbegin()[1].stack.size() < operationStackRequireArr[opIdx]) {
//            print("{:<50}{}\n", "Skipping program that underflows", serializeProgram(frameVec));
            // Generate the next variation while staying on the same level. This causes a move away from the current
            // program, which prevents it from becoming the root for any new programs, all of which would be invalid.
            continue;
        }
        frameVec.back() = nextFrame(frameVec.rbegin()[1], frameVec.back().op);
        if (frameVec.back().stack.size() != 1) {
//            print("{:<50}{} results={}\n", "Skipping program that returns <> 1 results", serializeProgram(frameVec), frameVec.back().stack.size());
            // Move to a lower level if possible. Programs that don't underflow but return a stack with more or less
            // than one item may become valid when adding one or more operations, so those branches must be searched.
            descendIfPossible = true;
            continue;
        }
//        print("{:<50}{}\n", "########## Returning valid program ##########", serializeProgram(frameVec));
        return true;
    }
}

Frame nextFrame(Frame frame, Operation operation)
{
    frame.op = operation;
    evalOperation(frame.stack, operation);
    return frame;
}

void evalOperation(BitsStack& s, const Operation operation)
{
    switch (operation) {
        case LD_A: s.push_back(0b0000000011111111);
            break;
        case LD_B: s.push_back(0b0000111100001111);
            break;
        case LD_C: s.push_back(0b0011001100110011);
            break;
        case LD_D: s.push_back(0b0101010101010101);
            break;
        case AND: s.push_back(pop(s) & pop(s));
            break;
        case OR: s.push_back(pop(s) | pop(s));
            break;
        case XOR: s.push_back(pop(s) ^ pop(s));
            break;
        case NOT: s.push_back(~pop(s));
            break;
        case ENUM_END: assert(false);
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
            f << format("{:016b}: {}\n", truthTable, serializeProgram(program));
        }
        ++truthTable;
    }
}

string serializeProgram(const FrameVec& frameVec)
{
    string s;
    for (auto frame : frameVec) {
        s += format("{} ", opStrVec[static_cast<int>(frame.op)]);
    }
    return s;
}

u64 pow(u32 base, u32 exp)
{
    u64 r = base;
    for (u32 i = 0; i < exp; ++i) {
        r *= base;
    }
    return r;
}
