// Style: http://geosoft.no/development/cppstyle.html

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stack>
#include <vector>

#include <cppformat/format.h>

#include "int_types.h"

using namespace std;
using namespace fmt;

// 8 operations
enum Operation  { LD_A, LD_B, LD_C, LD_D, AND, OR, XOR, NOT, ENUM_END };
const s32 operationStackDiffArr[]{1, 1, 1, 1, -1, -1, -1, 0};
const s32 operationStackRequireArr[]{0, 0, 0, 0, 2, 2, 2, 1};
const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "and", "or", "xor", "not", "ENUM_END"};
// 12 operations
//enum Operation {
//    LD_A, LD_B, LD_C, LD_D, LD_ONES, LD_ZEROS, AND, OR, XOR, NOT, DUP, POP, ENUM_END
//};
//const s32 operationStackDiffArr[]{1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 1, -1};
//const s32 operationStackRequireArr[]{0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1};
//const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "ones", "zeros", "and", "or", "xor", "not", "dup", "pop", "ENUM_END"};

typedef vector<Operation> OperationVec;
typedef vector<OperationVec> ProgramVec;
typedef u16 Bits;
typedef stack<Bits> BitsStack;
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

u64 countValidPrograms(u32 nSearchLevels);
void testProgramGenerator(u32 nSearchLevels);
bool nextValidProgram(OperationVec& program, u32 nSearchLevels);
inline bool stackUnderflows(bool& hasOneStackItem, const OperationVec& program);
Bits evalProgram(const OperationVec& program);
void evalOperation(BitsStack& s, Operation operation);
inline Bits pop(BitsStack& s);
void writeResults(const string& resultPath, const ProgramVec& programVec);
string serializeProgram(const OperationVec& program);
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
    print("Total valid programs: {}\n", countValidPrograms(nSearchLevels));

    ProgramVec foundProgramVec(nTotalTruthTables);
    u32 nFoundTruthTables = 0;
    #pragma omp parallel for
    for (auto opIdx = static_cast<int>(LD_A); opIdx <= static_cast<int>(LD_D); opIdx++) {
        OperationVec program = { static_cast<Operation>(opIdx) };
        u64 nProgramsEvaluated = 0;
        auto startTime = Time::now();
        #pragma omp critical
        {
            print("Starting thread {}/{} with program: {}\n", omp_get_thread_num(), omp_get_num_threads(),
                  serializeProgram(program));
        }
        while (true) {
            Bits truthTable = evalProgram(program);
            if (!foundProgramVec[truthTable].size() || foundProgramVec[truthTable].size() > program.size()) {
                #pragma omp critical
                {
                    if (!foundProgramVec[truthTable].size()) {
                        ++nFoundTruthTables;
                    }
//                    print("{} {:016b}: {}\n", opIdx, truthTable, serializeProgram(program));
                    foundProgramVec[truthTable] = program;
                }
            }
            ++nProgramsEvaluated;
            if ((!(nProgramsEvaluated & 0xffffff)) || nFoundTruthTables == nTotalTruthTables) {
                #pragma omp critical
                {
                    Fsec elapsedSec = Time::now() - startTime;
                    print("Thread: {}\n", opIdx);
                    print("Walltime, this thread: {:.2f}s\n", elapsedSec.count());
                    print("Evaluated, this thread: {} ({:d} per sec)\n", nProgramsEvaluated,
                          static_cast<u32>(nProgramsEvaluated / elapsedSec.count()));
                    print("Truth tables, all threads: {} ({:.2f}%)\n", nFoundTruthTables,
                          static_cast<float>(nFoundTruthTables) / nTotalTruthTables * 100.0f);
                    print("Operations in program: {}\n\n", program.size());
                }
            }
            if (!nextValidProgram(program, nSearchLevels)) {
                break;
            }
        }
    }
    writeResults("bitwise.txt", foundProgramVec);
//    getchar();
    return 0;
}

u64 countValidPrograms(u32 nSearchLevels)
{
    u64 nValidPrograms = 1;
    OperationVec program = { LD_A };
    while (nextValidProgram(program, nSearchLevels)) {
        ++nValidPrograms;
        if (!(nValidPrograms & 0xffffff)) {
            print("{}...\n", nValidPrograms);
        }
    }
    return nValidPrograms;
}

void testProgramGenerator(u32 nSearchLevels)
{
    OperationVec program = { LD_A };
    while (nextValidProgram(program, nSearchLevels));
}

// Given a current program, generate the next valid program. Return false when there are no more programs.
bool nextValidProgram(OperationVec& program, u32 nSearchLevels)
{
    bool descendIfPossible = true;
    while (true) {
        // When the function is called, we start by checking if it's possible to descend one step down.
        // If that's possible, we descend and return the program if it's valid.
        if (descendIfPossible && program.size() < nSearchLevels) {
            program.push_back(LD_A);
        }
        else {
            auto newOpIdx = static_cast<int>(program.back()) + 1;
            program.back() = static_cast<Operation>(newOpIdx);
            if (program.back() == ENUM_END) {
                // We have iterated over all possible variations at this level. We drop back to a higher level.
                program.pop_back();
                if (!program.size()) {
//                    print("{:<50}{}\n", "Program is empty. Iteration is complete", serializeProgram(program));
                    return false;
                }
                else {
//                    print("{:<50}{}\n", "Ascend to higher level, checked earlier", serializeProgram(program));
                    // We're back to a level that has already been returned. We jump to the top to generate the next
                    // variation on the same level. If the new variation passes the tests, it is returned as the next
                    // valid program and then becomes the root for further programs.
                    descendIfPossible = false;
                    continue;
                }
            }
        }
        bool hasOneStackItem;
        bool hasUnderflow = stackUnderflows(hasOneStackItem, program);
        if (hasUnderflow) {
//            print("{:<50}{}\n", "Skipping program that underflows", serializeProgram(program));
            // Generate the next variation while staying on the same level. This causes a move away from the current
            // program, which prevents it from becoming the root for any new programs, all of which would be invalid.
            descendIfPossible = false;
            continue;
        }
        if (!hasOneStackItem) {
//            print("{:<50}{}\n", "Skipping program that returns <> 1 results", serializeProgram(program));
            // Move to a lower level if possible. Programs that don't underflow but return a stack with more or less
            // than one item may become valid when adding one or more operations, so those branches must be searched.
            descendIfPossible = true;
            continue;
        }
//        print("{:<50}{}\n", "########## Returning valid program ##########", serializeProgram(program));
        return true;
    }
}

// For a program to be valid, it must not cause a stack underflow during evaluation and must leave exactly one value on
// the stack when completed. Returns true if stack underflows and sets {hasOneStackItem}.
bool stackUnderflows(bool& hasOneStackItem, const OperationVec& program)
{
    s32 nStackItems = 0;
    for (auto operation : program) {
        auto opIdx = static_cast<int>(operation);
        if (nStackItems < operationStackRequireArr[opIdx]) {
            return true;
        }
        nStackItems += operationStackDiffArr[opIdx];
    }
    hasOneStackItem = nStackItems == 1;
    return false;
}

// Find the truth table for a given program.
Bits evalProgram(const OperationVec& program)
{
    BitsStack s;
    for (auto operation : program) {
        evalOperation(s, operation);
    }
    return s.top();
}

void evalOperation(BitsStack& s, Operation operation)
{
    switch (operation) {
        case LD_A: s.push(0b0000000011111111);
            break;
        case LD_B: s.push(0b0000111100001111);
            break;
        case LD_C: s.push(0b0011001100110011);
            break;
        case LD_D: s.push(0b0101010101010101);
            break;
//        case LD_ONES: s.push(0xffff);
//            break;
//        case LD_ZEROS: s.push(0x0000);
//            break;
        case AND: s.push(pop(s) & pop(s));
            break;
        case OR: s.push(pop(s) | pop(s));
            break;
        case XOR: s.push(pop(s) ^ pop(s));
            break;
        case NOT: s.push(~pop(s));
            break;
//        case DUP: s.push(s.top());
//            break;
//        case POP: s.pop();
//            break;
        case ENUM_END: assert(false);
            break;
        default: assert(false);
            break;
    }
}

Bits pop(BitsStack& s)
{
    Bits v = s.top();
    s.pop();
    return v;
}

void writeResults(const string& resultPath, const ProgramVec& programVec)
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

string serializeProgram(const OperationVec& program)
{
    string s;
    for (auto operation : program) {
        s += format("{} ", opStrVec[static_cast<int>(operation)]);
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
