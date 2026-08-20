#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#ifndef RUNTIME
#define RUNTIME -1
#endif

using namespace std;
#define MAXN 133
#define MAXM 3333
int M[MAXN][MAXN];
int D[MAXN][MAXM], C[MAXM][MAXM];
struct ProgramSeq { int t, a, b, y; } Program[MAXM];
int n, m, lenProg;

int bestX[MAXM], bestY[MAXM], bestUsed[MAXM], bestCount;
// int finalArr[MAXM];


/*if you use Linux please remove the function below*/
class MT19937
{
private:
    unsigned state[624];
    unsigned index;
public:
    MT19937()
    {
        srand(time(0));
        for(int i=0;i<624;i++) state[i]=(rand()<<30)|(rand()<<15)|rand();
    }
    void twist()
    {
        for(int i=0;i<624;i++)
        {
            unsigned y=(state[i]&0x80000000u)|(state[(i+1)%624]&0x7fffffff);
            state[i] = (y >> 1) ^ state[(i + 397) % 624];
            if(!(y&1))
                state[i]^=0x9908b0df;
        }
    }
    unsigned getstate()
    {
        if(index==0) twist();
        int y = state[index];
        y = y ^ y >> 11;
        y = y ^ y << 7 & 2636928640u;
        y = y ^ y << 15 & 4022730752u;
        y = y ^ y >> 18;
        index=(index+1)%624;
        return y;
    }
};
MT19937 *mtt;

void reset_globals() {
    memset(D, 0, sizeof(D));
    memset(C, 0, sizeof(C));
    // memset(finalArr, 0, sizeof(finalArr));
    m = n;
    lenProg = 0;
    bestCount = 0;
}

__global__ void compute_C_kernel(int* Dx, int* Cx,int nx,int mx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("%d %d %d\n",j,k,mx);
    if (j >= mx || k >= mx || j >= k) return;

    int count = 0;
    for (int i = 0; i < nx; i++) {
        if (Dx[i * mx + j] && Dx[i * mx + k]) {
            count++;
        }
    }
    Cx[j * mx + k] = count;  
}

void getC_gpu() {
    int *Dgpu,*Cgpu;
    cudaMalloc(&Dgpu,4*n*m);
    cudaMalloc(&Cgpu,4*m*m);
    int *Dflat=(int*)malloc(4*n*m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            Dflat[i*m+j]=D[i][j];
    
    cudaMemcpy(Dgpu,Dflat,4*n*m,cudaMemcpyHostToDevice);
    free(Dflat);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + 15) / 16, (m + 15) / 16);
    compute_C_kernel<<<blocksPerGrid, threadsPerBlock>>>(Dgpu, Cgpu,n,m);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *Cflat=(int*)malloc(4*m*m);
    cudaMemcpy(Cflat,Cgpu,4*m*m,cudaMemcpyDeviceToHost);
    for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
            C[i][j]=Cflat[i*m+j];
    free(Cflat);

    cudaFree(Dgpu);
    cudaFree(Cgpu);
}

void deal() {
    reset_globals();
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            D[i][j] = M[i][j];

    while(1) {
        if(lenProg > 3330) return;
        getC_gpu();
        
        int Cmax = 0;
        bestCount = 0;
        for(int i = 0; i < m; i++) {
            for(int j = i + 1; j < m; j++) {
                if(C[i][j] > Cmax) {
                    Cmax = C[i][j];
                    bestCount = 1;
                    bestX[0] = i;
                    bestY[0] = j;
                    bestUsed[0] = 0;
                } else if(C[i][j] == Cmax) {
                    bestX[bestCount] = i;
                    bestY[bestCount] = j;
                    bestUsed[bestCount] = 0;
                    ++bestCount;
                }
            }
        }
        if(Cmax <= 1) break;
        
        while(bestCount)
        {
            int useIndex=(mtt->getstate())%bestCount;
            int bx=bestX[useIndex],by=bestY[useIndex];
            bestUsed[useIndex]=1;
            Program[lenProg++]={m,bx,by,-1};
            for(int i=0;i<n;i++)
                if(D[i][bx]&&D[i][by])
                {
                    D[i][bx]=0;
                    D[i][by]=0;
                    D[i][m]=1;
                }
            ++m;
            for(int i=0;i<bestCount;i++)
            {
                if(i==useIndex) continue;
                if(bestX[i]-bx&&bestY[i]-bx&&bestX[i]-by&&bestY[i]-by) continue;
                int ex=bestX[i],ey=bestY[i],eCount=0;
                for(int j=0;j<n;j++)
					eCount+=(D[j][ex]&&D[j][ey]);
				if(eCount-Cmax) bestUsed[i]=1;
            }
            for(int i=0;i<m-1;i++)
            {
                int eCount=0;
				for(int j=0;j<n;j++)
					eCount+=(D[j][i]&D[j][m-1]);
				if(eCount==Cmax)
				{
					bestX[bestCount]=m-1;
					bestY[bestCount]=i;
					bestUsed[bestCount]=0;
					++bestCount;
				}
            }
            int cur=0;
			for(int i=0;i<bestCount;i++)
			{
				if(!bestUsed[i])
				{
					bestX[cur]=bestX[i];
					bestY[cur]=bestY[i];
					bestUsed[cur]=0;
					++cur;
				}
			}
			bestCount=cur;
			if(!bestCount) break;
        }
    }
    
    int current_m = m;
    for(int i = 0; i < n; i++) {
        vector<int> ones;
        for(int j = 0; j < m; j++)
            if(D[i][j]) ones.push_back(j);
        
        if(ones.size() == 2) {
            Program[lenProg++] = {current_m++, ones[0], ones[1], i};
        } else if(ones.size() > 2) {
            int current = ones[0];
            for(size_t j = 1; j < ones.size() - 1; j++) {
                Program[lenProg++] = {current_m, current, ones[j], -1};
                current = current_m++;
            }
            Program[lenProg++] = {current_m, current, ones.back(), i};
            current_m++;
        }
    }
}

int main(int argc, char** argv) {
    if(argc < 3) {
        cerr << "Usage: " << argv[0] << " <input> <output>" << endl;
        return 1;
    }
    
    srand(time(0));
    mtt=new MT19937();
    FILE* fp = fopen(argv[1], "r");
    if(!fp) {
        cerr << "Error opening input file" << endl;
        return 1;
    }
    
    fscanf(fp, "%d", &n);
    int SIZE=n;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            fscanf(fp, "%d", &M[i][j]);
    fclose(fp);
    int BEST = 0x77777777,NTH=0;
    unsigned runtime_left=RUNTIME;
    while(runtime_left--) {
        n=SIZE;
        m=SIZE;
        deal();  
        cout<<(NTH++)<<" "<<lenProg<<endl;
        if(lenProg < BEST) {
            BEST = lenProg;
            cout << "New best: " << lenProg << endl;
            
            fp = fopen(argv[2], "w");
            if(!fp) {
                cerr << "Error opening output file" << endl;
                continue;
            }
            
            fprintf(fp, "%d %d\n",SIZE, BEST);
            for(int i = 0; i < BEST; i++) 
                fprintf(fp,"%d %d %d %d\n",Program[i].t, Program[i].a, Program[i].b,Program[i].y);
            
            fclose(fp);
        }
    }
    return 0;
}
