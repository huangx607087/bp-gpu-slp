
#include <math.h>
#include <ctype.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

const int MaxBaseSize = 1000;
const bool PRINTROWS = true;

int NumInputs;
int NumTargets;
int ProgramSize;
long long int Target[MaxBaseSize];
int Dist[MaxBaseSize];    //distance from current base to Target[i]
int NDist[MaxBaseSize]; //what Dist would be if NewBase was added
long long int Base[MaxBaseSize];
int BaseSize;
int TargetsFound;
unsigned long long Program[2000];//Added to record Program.
void InitBase();
void ReadTargetMatrix();
bool is_target(long long int x);
bool is_base(long long int x);
int NewDistance(int u); //calculates the distance from the base to Target[u]
int TotalDistance();//returns the sum of distances to targets
bool reachable(long long int T, int K, int S);
bool EasyMove(); //if any two bases add up to a target, pick them
void PickNewBaseElement();
void binprint(long long int x); //outputs last NumInputs bits of x

ifstream TheMatrix;
//ofstream out_file;

int main(int argc, char *argv[])
{
    int NumMatrices=1;

    clock_t t1 = clock();
    TheMatrix.open(argv[1]);

    // TheMatrix >> NumMatrices;
    for (int i = 0; i < NumMatrices; i++)
    {
        ReadTargetMatrix();
        InitBase();
        ProgramSize = 0;
        int counter = 0;
        while (TargetsFound < NumTargets)
        {
            counter++;
            if (!EasyMove())
                PickNewBaseElement();
        }
        cout << ProgramSize << endl;
    }

    char FILENAME[40];
    time_t timestamp;
    time(&timestamp);
    sprintf(FILENAME,"Result_%s_%06x.txt",argv[1],timestamp&0xffffff);
    FILE *fp=fopen(FILENAME,"w");
    clock_t t2 = clock();
    cout << (t2 - t1) / (double)CLOCKS_PER_SEC<<endl;
    fprintf(fp,"Total Time Cost(s):%lf\n", (t2 - t1) / (double)CLOCKS_PER_SEC);
    for(int i=0;i<ProgramSize;i++)
    {
        unsigned long long x=Program[i];
        short buffer[4];
        for(int j=0;j<4;j++)
        {
            buffer[j]=x&0xffff;
            x>>=16;
        }
        fprintf(fp,"%d %d %d %d\n",buffer[3],buffer[2],buffer[1],buffer[0]);
    }
    fclose(fp);
} //main

void InitBase()
{
    TargetsFound = 0;
    Base[0] = 1;
    for (int i = 1; i < NumInputs; i++)
        Base[i] = 2 * Base[i - 1];
    BaseSize = NumInputs; //initial base is just the xi's
    for (int i = 0; i < NumTargets; i++)
        if (Dist[i] == 0)
            TargetsFound++;
}

int TotalDistance() //returns the sum of distances to targets
{
    int D = 0;
    int t;
    for (int i = 0; i < NumTargets; i++)
    {
        t = NewDistance(i);
        NDist[i] = t;
        D = D + t;
    }
    return D;
}

long long int NewBase; //global variable containing a candidate new base

bool EasyMove()
{
    int t;
    bool foundone = false;

    //see if anything in the distance vector is 1
    for (int i = 0; i < NumTargets; i++)
        if (Dist[i] == 1)
        {
            foundone = true;
            t = i;
            break;
        }
    if (!foundone)
        return false;
    //update Dist array
    int besti,bestj,notFound=1;
    for(int i=0;i<BaseSize&&notFound;i++)
    {
        for(int j=1+i;j<BaseSize&&notFound;j++)
            if(Base[i]^Base[j]==Target[t])
            {
                besti=i;
                bestj=j;
                notFound=0;
            }
    }
    NewBase = Target[t];
    for (int u = 0; u < NumTargets; u++)
        Dist[u] = NewDistance(u);
    //update Base with NewBase
    Base[BaseSize] = NewBase;
    
    Program[ProgramSize]=(BaseSize<<16)|(besti);
    Program[ProgramSize]<<=32;
    Program[ProgramSize]|=(bestj<<16)|(t);
    BaseSize++;
    ProgramSize++;
    TargetsFound++;
    return true;
} //EasyMove()

/* PickNewBaseElement is only called when there are no 1's in Dist[]*/
void PickNewBaseElement()
{
    int MinDistance;
    long long int TheBest;
    int ThisDist;
    int ThisNorm, OldNorm;
    int besti, bestj, d;
    bool easytarget;
    int BestDist[MaxBaseSize];

    MinDistance = BaseSize * NumTargets; //i.e. something big
    OldNorm = 0;                                                 //i.e. something small
    //try all pairs of bases
    for (int i = 0; i < BaseSize - 1; i++)
    {
        for (int j = i + 1; j < BaseSize; j++)
        {
            NewBase = Base[i] ^ Base[j];
            //sanity check
            if (NewBase == 0)
            {
                cout << "a base is 0, should't happen " << endl;
                exit(0);
            }
            //if NewBase is not new continue
            if (is_base(NewBase))
                continue;
            //if NewBase is target then choose it
            easytarget = false;
            if (is_target(NewBase))
            {
                cout << "shouldn't find an easy target here " << endl;
                exit(0);
                easytarget = true;
                besti = i;
                bestj = j;
                TheBest = NewBase;
                break;
            }
            ThisDist = TotalDistance(); //this also calculates NDist[]
            if (ThisDist <= MinDistance)
            {
                //calculate Norm
                ThisNorm = 0;
                for (int k = 0; k < NumTargets; k++)
                {
                    d = NDist[k];
                    ThisNorm = ThisNorm + d * d;
                }
                //resolve tie in favor of largest norm
                if ((ThisDist < MinDistance) || (ThisNorm > OldNorm))
                {
                    besti = i;
                    bestj = j;
                    TheBest = NewBase;
                    for (int uu = 0; uu < NumTargets; uu++)
                        BestDist[uu] = NDist[uu];
                    MinDistance = ThisDist;
                    OldNorm = ThisNorm;
                }
            }
        }
        if (easytarget)
            break;
    }
    //update Dist array
    NewBase = TheBest;
    for (int i = 0; i < NumTargets; i++)
        Dist[i] = BestDist[i];
    //update Base with TheBest
    Base[BaseSize] = TheBest;
    
    Program[ProgramSize]=(BaseSize<<16)|(besti);
    Program[ProgramSize]<<=32;
    Program[ProgramSize]|=(bestj<<16)|(65535);
    BaseSize++;
    ProgramSize++;
    if (is_target(TheBest))
        TargetsFound++; //this shouldn't happen
} //PickNewBaseElement()

void binprint(long long int x) //outputs last NumInputs bits of x
{
    long long int t = x;
    for (int i = 0; i < NumInputs; i++)
    {
        if (t % 2)
            cout << "1 ";
        else
            cout << "0 ";
        t = t / 2;
    }
} //binprint

void ReadTargetMatrix()
{
    TheMatrix >> NumTargets;
    NumInputs = NumTargets;
    // TheMatrix >> NumInputs;
    //check that NumInputs is < wordsize
    if (NumInputs >= 8 * sizeof(long long int))
    {
        cout << "too many inputs" << endl;
        exit(0);
    }

    int bit;
    for (int i = 0; i < NumTargets; i++)
    //read row i
    {
        long long int PowerOfTwo = 1;
        Target[i] = 0;
        Dist[i] = -1; //initial distance from Target[i] is Hamming weight - 1
        for (int j = 0; j < NumInputs; j++)
        {
            TheMatrix >> bit;
            if (bit)
            {
                Dist[i]++;
                Target[i] = Target[i] + PowerOfTwo;
            }
            PowerOfTwo = PowerOfTwo * 2;
        }
    }
} //ReadTargetMatrix()

bool is_target(long long int x)
{
    for (int i = 0; i < NumTargets; i++)
        if (x == Target[i])
            return true;
    return false;
} //is_target

bool is_base(long long int x)
{
    //sanity check, shouldn't ask if 0 is base
    if (x == 0)
    {
        cout << "asking if 0 is in Base " << endl;
        exit(0);
    }

    for (int i = 0; i < BaseSize; i++)
        if (x == Base[i])
            return true;
    return false;
} //is_base

// Distance is 1 less than the number of elements
// in the base that I need to add in order to get Target[u].
// The next function calculates the distance from the base,
// augmented by NewBase, to Target[u]. Uses the following observations:
// Adding to the base can only decrease distance.
// Also, since NewBase is the sum of two old base
// elements, the distance from the augmented base
// to Target[u] can decrease at most by 1. If the
// the distance decreases, then NewBase must be one
// of the summands.

int NewDistance(int u)
{
    //if Target[u] is in augmented base return 0;
    if (is_base(Target[u]) || (NewBase == Target[u]))
        return 0;

    // Try all combinations of Dist[u]-1 base elements until one sums
    // to Target[u] + NewBase. If this is true, then Target[u] is the
    // sum of Dist[u] elements in the augmented base, and therefore
    // the distance decreases by 1.

    if (reachable(Target[u] ^ NewBase, Dist[u] - 1, 0))
        return (Dist[u] - 1);
    else
        return Dist[u]; //keep old distance
} //NewDistance(int u)

//return true if T is the sum of K elements among Base[S..BaseSize-1]
bool reachable(long long int T, int K, int S)
{
    if ((BaseSize - S) < K)
        return false; //not enough base elements

    if (K == 0)
        return false; //this is probably not reached
    if (K == 1)
    {
        for (int i = S; i < BaseSize; i++)
            if (T == Base[i])
                return true;
        return false;
    }

    //consider those sums containing Base[S]
    if (reachable(T ^ Base[S], K - 1, S + 1))
        return true;
    //consider those sums not containing Base[S]
    if (reachable(T, K, S + 1))
        return true;
    //not found
    return false;
} // reachable(long long int T, int K, int S)
