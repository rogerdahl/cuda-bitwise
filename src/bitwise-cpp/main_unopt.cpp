// Style: http://geosoft.no/development/cppstyle.html

#include <chrono>
#include <iostream>
#include <stack>
#include <vector>

#include <cppformat/format.h>

#include "int_types.h"

using namespace std;
using namespace fmt;

// The first version of this program would take 2500 hours to run. This was reduced to 250 hours by changing the stack
// type from std::stack to std::vector and clearing and reusing the std::vector for each program evaluation. The
// std::stack could not be cleared and therefore had to be recreated for each program evaluation.

enum Operation {
    LD_A, LD_B, LD_C, LD_D, AND, OR, XOR, NOT, ENUM_END
};
const vector<string> opStrVec = {"v1", "v2", "v3", "v4", "and", "or", "xor", "not", "ENUM_END"};

typedef vector<Operation> OperationVec;
typedef vector<u16> U16Stack;
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> Fsec;

void nextProgram(OperationVec& program);
bool evalProgram(u16& truthTable, const OperationVec& program);
bool evalOperation(U16Stack& s, Operation operation);
u16 pop(U16Stack& s);
string serializeProgram(const OperationVec& program);


int main() {
    const u32 nSearchLevels = 15;
    const u64 nTotalPossiblePrograms = pow(static_cast<int>(ENUM_END), nSearchLevels);
    const u32 nPossibleTruthTables = 1 << 16;
    // 1 << 16 to cover the full search space. Lower values for testing and benchmarking.
    const u32 nSearchTruthTables = 1 << 16;

    print("Total possible programs: {}\n", nTotalPossiblePrograms);

    u32 nFoundTruthTables = 0;
    u64 nProgramsEvaluated = 0;
    u64 nValidPrograms = 0;
    OperationVec program = {LD_A};
    vector<OperationVec> foundOperationVec(nPossibleTruthTables);
    auto startTime = Time::now();

    while (nFoundTruthTables < nSearchTruthTables) {
        u16 truthTable;
        bool isValidProgram = evalProgram(truthTable, program);
        if (isValidProgram) {
            ++nValidPrograms;
            if (!foundOperationVec[truthTable].size()) {
                foundOperationVec[truthTable] = program;
                ++nFoundTruthTables;
            }
        }
        ++nProgramsEvaluated;

        if (!(nProgramsEvaluated & 0xffffff) || nFoundTruthTables == nSearchTruthTables) {
            Fsec elapsedSec = Time::now() - startTime;
            print("\nTotal time: {:.2f}s\n", elapsedSec.count());
            print("Hours until done: {:.2f}\n", static_cast<float>(nTotalPossiblePrograms) / nProgramsEvaluated * elapsedSec.count() / 60.0f / 60.0f);
            print("Programs evaluated: {} ({:f}%)\n", nProgramsEvaluated, static_cast<float>(nProgramsEvaluated) / nTotalPossiblePrograms * 100.0f);
            print("Programs evaluated per second: {:d}\n", static_cast<int>(nProgramsEvaluated / elapsedSec.count()));
            print("Valid programs: {} ({:.2f}%)\n", nValidPrograms, static_cast<float>(nValidPrograms) / nProgramsEvaluated * 100.0f);
            print("Found truth tables: {} ({:.2f}%)\n", nFoundTruthTables, static_cast<float>(nFoundTruthTables) / nSearchTruthTables * 100.0f);
            print("Last evaluated: {} ({} ops)\n", serializeProgram(program), program.size());
        }

        nextProgram(program);
    }

    return 0;
}

void nextProgram(OperationVec& program) {
    auto nOperations = program.size();
    for (u32 i = 0; i < nOperations; ++i) {
        program[i] = static_cast<Operation>(static_cast<int>(program[i]) + 1);
        if (program[i] != ENUM_END) {
            break;
        }
        program[i] = LD_A;
        if (i == nOperations - 1) {
            program.push_back(LD_A);
        }
    }
}

// Find the truth table for a given program.
// Return true for valid programs. For a program to be valid, it must not cause a stack underflow during evaluation
// and must leave exactly one value on the stack when completed.
bool evalProgram(u16& truthTable, const OperationVec& program) {
    static U16Stack s;
    s.clear();

    for (auto operation : program) {
        bool stackIsValid = evalOperation(s, operation);
        if (!stackIsValid) {
            return false;
        }
    }
    if (s.size() != 1) {
        return false;
    }
    truthTable = s.back();
    return true;
}

bool evalOperation(U16Stack& s, Operation operation) {
    switch (operation) {
        case LD_A:
            s.push_back(0b0000000011111111);
            break;
        case LD_B:
            s.push_back(0b0000111100001111);
            break;
        case LD_C:
            s.push_back(0b0011001100110011);
            break;
        case LD_D:
            s.push_back(0b0101010101010101);
            break;
        case AND:
            if (s.size() < 2) return false;
            s.push_back(pop(s) & pop(s));
            break;
        case OR:
            if (s.size() < 2) return false;
            s.push_back(pop(s) | pop(s));
            break;
        case XOR:
            if (s.size() < 2) return false;
            s.push_back(pop(s) ^ pop(s));
            break;
        case NOT:
            if (!s.size()) return false;
            s.push_back(~pop(s));
            break;
        case ENUM_END:
            assert(false);
    }
    return true;
}

u16 pop(U16Stack& s) {
    u16 v = s.back();
    s.pop_back();
    return v;
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
