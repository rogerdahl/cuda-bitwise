// Style: http://geosoft.no/development/cppstyle.html

#include <chrono>
#include <fstream>
#include <iostream>
#include <stack>
#include <vector>

#include <cppformat/format.h>

#include "int_types.h"

using namespace std;
using namespace fmt;

enum Operation
{
    LD_A, LD_B, LD_C, LD_D, LD_ONES, LD_ZEROS, AND, OR, XOR, NOT, DUP, POP, ENUM_END
};
const s32 operationStackDiffArr[]{1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 1, -1};
const s32 operationStackRequireArr[]{0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1};

typedef vector<Operation> OperationVec;
typedef vector<OperationVec> ProgramVec;
typedef u16 Bits;
typedef stack<Bits> BitsStack;
typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Fsec;

void nextProgram(OperationVec& program);
void nextValidProgram(OperationVec& program, u32 nSearchLevels);
bool isValidProgram(const OperationVec& program);
Bits evalProgram(const OperationVec& program);
void evalOperation(BitsStack& s, Operation operation);
inline Bits pop(BitsStack& s);
void writeResults(const string& resultPath, const ProgramVec& programVec);
string serializeProgram(const OperationVec& program);
bool stackUnderflows(const OperationVec& program);

int main()
{
//    OperationVec program = {LD_A};
//    while (true) {
//        print("Ok: {}\n", serializeProgram(program));
//        nextValidProgram(program, 5);
//        if (!program.size()) {
//            break;
//        }
//    }

    const u32 numPossibleTruthTables = 1 << 16;
//    const u32 numSearchTruthTables = 500;
    const u32 numSearchTruthTables = 1 << 16; // Cover the full search space
    u32 numFoundTruthTables = 0;
    u64 numProgramsEvaluated = 0;
    u64 numValidPrograms = 0;
    OperationVec program = {LD_A};
    ProgramVec foundProgramVec(numPossibleTruthTables);
    auto startTime = Time::now();

    int nSearchLevels = 12;

    while (numFoundTruthTables < numSearchTruthTables && program.size()) {
        if (isValidProgram(program)) {
//            print("Valid\n");
            ++numValidPrograms;
            Bits truthTable = evalProgram(program);
            if (!foundProgramVec[truthTable].size() > program.size()) {
                print("Found: {:016b}: {}\n", truthTable, serializeProgram(program));
                foundProgramVec[truthTable] = program;
                ++numFoundTruthTables;
            }
        }

        ++numProgramsEvaluated;

        // Both using mod and checking for elapsed time here (to determine if status should be printed) was expensive.
        if ((!(numProgramsEvaluated & 0xffffff)) || numFoundTruthTables == numSearchTruthTables) {
            Fsec elapsedSec = Time::now() - startTime;
            print("Total time: {:.2f}s\n", elapsedSec.count());
            print("Programs evaluated: {}\n", numProgramsEvaluated);
            print("Programs evaluated per second: {:d}\n", static_cast<int>(numProgramsEvaluated / elapsedSec.count()));
            print("Valid programs: {} ({:.2f}%)\n", numValidPrograms,
                  static_cast<float>(numValidPrograms) / numProgramsEvaluated * 100.0f);
            print("Found truth tables: {} ({:.2f}%)\n", numFoundTruthTables,
                  static_cast<float>(numFoundTruthTables) / numSearchTruthTables * 100.0f);
            print("Operations in program: {}\n\n", program.size());
        }

        nextValidProgram(program, nSearchLevels);
    }

    writeResults("bitwise.txt", foundProgramVec);

//    getchar();
    return 0;
}

void nextValidProgram(OperationVec& program, u32 nSearchLevels)
{
    if (program.size() < NUM_SEARCH_LEVELS) {
        program.push_back(LD_A);
        return;
    }
    while (true) {
        auto newOpIdx = static_cast<int>(program.back()) + 1;
        program.back() = static_cast<Operation>(newOpIdx);
        if (program.back() != ENUM_END && !stackUnderflows(program)) {
            return;
        }
//        print("Rejected: {}\n", serializeProgram(program));
        program.pop_back();
        if (!program.size()) {
            return;
        }
    }
}

//void nextProgram(OperationVec& program)
//{
//    auto numOperations = program.size();
//    for (u32 i = 0; i < numOperations; ++i) {
//        program[i] = static_cast<Operation>(static_cast<int>(program[i]) + 1);
//        if (program[i] != ENUM_END) {
//            break;
//        }
//        program[i] = LD_A;
//        if (i == numOperations - 1) {
//            program.push_back(LD_A);
//        }
//    }
//}

bool stackUnderflows(const OperationVec& program)
{
//    print("{:016b}: {}\n", 0, serializeProgram(program));
    s32 numStackItems = 0;
    for (auto operation : program) {
        u32 i = static_cast<u32>(operation);
        if (numStackItems < operationStackRequireArr[i]) {
            return true;
        }
        numStackItems += operationStackDiffArr[i];
    }
    return false;
}

// Return true for valid programs. For a program to be valid, it must not cause a stack underflow during evaluation
// and must leave exactly one value on the stack when completed.
bool isValidProgram(const OperationVec& program)
{
//    print("{:016b}: {}\n", 0, serializeProgram(program));
    s32 numStackItems = 0;
    for (auto operation : program) {
        u32 i = static_cast<u32>(operation);
        if (numStackItems < operationStackRequireArr[i]) {
            return false;
        }
        numStackItems += operationStackDiffArr[i];
    }
    return numStackItems == 1;
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
        case LD_ONES: s.push(0xffff);
            break;
        case LD_ZEROS: s.push(0x0000);
            break;
        case AND: s.push(pop(s) & pop(s));
            break;
        case OR: s.push(pop(s) | pop(s));
            break;
        case XOR: s.push(pop(s) ^ pop(s));
            break;
        case NOT: s.push(~pop(s));
            break;
        case DUP: s.push(s.top());
            break;
        case POP: s.pop();
            break;
        case ENUM_END: assert(false);
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
    vector<string> opVec = {"v1", "v2", "v3", "v4", "ones", "zeros", "and", "or", "xor", "not", "dup", "pop", "ENUM_END"};
    for (auto operation : program) {
        s += format("{} ", opVec[static_cast<int>(operation)]);
    }
    return s;
}
