#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>
using namespace std;
bool isFullRank(int n,vector<vector<uint8_t>>& matrix)
{
    vector<vector<uint8_t>> temp=matrix;
    int rank=0;
    vector<bool> row_used(n,false);
    for(int col=0;col<n;++col)
    {
        int pivot=-1;
        for(int row=0;row<n;++row)
        {
            if(!row_used[row]&&temp[row][col])
            {
                pivot=row;
                break;
            }
        }
        if(pivot+1)
        {
            rank++;
            row_used[pivot]=true;
            for(int row=0;row<n;++row)
                if(row!=pivot&&temp[row][col])
                    for(int k=col;k<n;++k)
                        temp[row][k]^=temp[pivot][k];
        }
    }
    return rank==n;
}
void generateRow(vector<uint8_t>& row,int n,int weight,mt19937& gen)
{
    fill(row.begin(),row.end(),0);
    vector<int> indices(n);
    iota(indices.begin(),indices.end(),0);
    shuffle(indices.begin(),indices.end(),gen);
    for(int i=0;i<weight;++i) row[indices[i]]=1;
}
int main(int argc,char* argv[])
{
    if(argc<3)
    {
        cerr<<"Usage: "<<argv[0]<<" n w [seed(default:1)]"<<endl;
        return 1;
    }
    int n=stoi(argv[1]);
    int w=stoi(argv[2]);
    unsigned int seed=(argc>=4)?stoi(argv[3]):1;
    mt19937 gen(seed);
    vector<vector<uint8_t>> matrix(n,vector<uint8_t>(n));
    int group_size=(n>>2);
    bool success=false;
    int trial=0;
    for(trial=1;trial<=4096;++trial)
    {
        for(int i=0;i<n;++i)
        {
            int current_w=w;
            if(i<group_size)current_w=w+1;
            else if(i<2*group_size)current_w=w-1;
            current_w=max(0,min(n,current_w));
            generateRow(matrix[i],n,current_w,gen);
        }
        if(isFullRank(n,matrix))
        {
            success=true;
            break;
        }
    }
    if(success)
    {
        char filename[100];
        sprintf(filename,"M%d_%d_%d.txt",n,w,seed);
        ofstream outfile(filename);
        outfile<<n<<"\n";
        for(int i=0;i<n;++i)
        {
            for(int j=0;j<n;++j)
                outfile<<(int)matrix[i][j]<<(j==n-1?"":" ");
            
            outfile<<"\n";
        }
        outfile.close();
        printf("Succeed generated within %d trials\n",trial);
    }
    else
        printf("Failed to generate within 4096 trials\n");
    return 0;
}