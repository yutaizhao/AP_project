//
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

//
typedef struct particle_s {
    
    f32 *restrict x, *restrict y, *restrict z;
    f32 *restrict vx, *restrict vy, *restrict vz;
    f32 *restrict fx, *restrict fy, *restrict fz;
    
} particle_t;

//
void init(particle_t *restrict p, u64 n)
{
    for (u64 i = 0; i < n; i++)
    {
        //
        u64 r1 = (u64)rand();
        u64 r2 = (u64)rand();
        f32 sign = (r1 > r2) ? 1 : -1;
        
        //
        p->x[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p->y[i] = (f32)rand() / (f32)RAND_MAX;
        p->z[i] = sign * (f32)rand() / (f32)RAND_MAX;
        
        //
        p->vx[i] = (f32)rand() / (f32)RAND_MAX;
        p->vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p->vz[i] = (f32)rand() / (f32)RAND_MAX;
        
        p->fx[i] = 0;
        p->fy[i] = 0;
        p->fz[i] = 0;
    }
}

//
void move_particles(particle_t *restrict p, const f32 dt, u64 n)
{
    //Used to avoid division by 0 when comparing a particle to itself
    const f32 softening = 1e-20;
    const u64 T = 64;
    
    //TFM provided by prof
#pragma omp parallel proc_bind(spread)
    {
#pragma omp for nowait
        //For all particles
        for (u64 i = 0; i < n; i++)
        {
            for (u64 j = 0; j < n; j+=T)
            {
                for (u64 jj = j; jj < fmin(j+T,n); jj++){
                    
                    const f32 d = 1/(( (p->x[j] - p->x[i])* (p->x[j] - p->x[i])) + ((p->y[j] - p->y[i]) * (p->y[j] - p->y[i])) + (( p->z[j] - p->z[i]) * ( p->z[j] - p->z[i])) + softening * sqrtf(( (p->x[j] - p->x[i])* (p->x[j] - p->x[i])) + ((p->y[j] - p->y[i]) * (p->y[j] - p->y[i])) + (( p->z[j] - p->z[i]) * ( p->z[j] - p->z[i])) + softening)); //12
                    
                    //Calculate net force: 6 FLOPs
                    p->fx[i] += (p->x[j] - p->x[i]) * d; //14 (add, mul)
                    p->fy[i] += (p->y[j] - p->y[i]) * d; //16 (add, mul)
                    p->fz[i] += (p->z[j] - p->z[i]) * d; //18 (add, mul)
                }
            }
            
        }
    }
    
    //Update positions: 6 FLOPs
    for (u64 i = 0; i < n; i+=4)
    {
        p->vx[i] += dt * p->fx[i]; //20 (mul, add)
        p->vy[i] += dt * p->fy[i]; //22 (mul, add)
        p->vz[i] += dt * p->fz[i]; //24 (mul, add)
        
        p->x[i] += dt * p->vx[i];
        p->y[i] += dt * p->vy[i];
        p->z[i] += dt * p->vz[i];
        
        p->vx[i+1] += dt * p->fx[i+1]; //20 (mul, add)
        p->vy[i+1] += dt * p->fy[i+1]; //22 (mul, add)
        p->vz[i+1] += dt * p->fz[i+1]; //24 (mul, add)

        p->x[i+1] += dt * p->vx[i+1];
        p->y[i+1] += dt * p->vy[i+1];
        p->z[i+1] += dt * p->vz[i+1];
        
        p->vx[i+2] += dt * p->fx[i+2]; //20 (mul, add)
        p->vy[i+2] += dt * p->fy[i+2]; //22 (mul, add)
        p->vz[i+2] += dt * p->fz[i+2]; //24 (mul, add)

        p->x[i+2] += dt * p->vx[i+2];
        p->y[i+2] += dt * p->vy[i+2];
        p->z[i+2] += dt * p->vz[i+2];
        
        p->vx[i+3] += dt * p->fx[i+3]; //20 (mul, add)
        p->vy[i+3] += dt * p->fy[i+3]; //22 (mul, add)
        p->vz[i+3] += dt * p->fz[i+3]; //24 (mul, add)

        p->x[i+3] += dt * p->vx[i+3];
        p->y[i+3] += dt * p->vy[i+3];
        p->z[i+3] += dt * p->vz[i+3];
        
        p->vx[i+4] += dt * p->fx[i+4]; //20 (mul, add)
        p->vy[i+4] += dt * p->fy[i+4]; //22 (mul, add)
        p->vz[i+4] += dt * p->fz[i+4]; //24 (mul, add)

        p->x[i+4] += dt * p->vx[i+4];
        p->y[i+4] += dt * p->vy[i+4];
        p->z[i+4] += dt * p->vz[i+4];
        
    }
}

//
int main(int argc, char **argv)
{
    
    u64 tn = 24;
    
    printf("threads = %d \n", tn);
    //Number of particles to simulate
    const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
    
    //Number of experiments
    const u64 steps= 13;
    
    //Time step
    const f32 dt = 0.01;
    
    //
    f64 rate = 0.0, drate = 0.0;
    
    //Steps to skip for warm up
    const u64 warmup = 3;
    
    u64 alignment = 32;
    //
    particle_t *restrict p = aligned_alloc(alignment,sizeof(particle_t));
    //
    p->x = aligned_alloc(alignment,sizeof(f32) * n);
    p->y = aligned_alloc(alignment,sizeof(f32) * n);
    p->z = aligned_alloc(alignment,sizeof(f32) * n);
    
    p->vx = aligned_alloc(alignment,sizeof(f32) * n);
    p->vy = aligned_alloc(alignment,sizeof(f32) * n);
    p->vz = aligned_alloc(alignment,sizeof(f32) * n);
    
    p->fx = aligned_alloc(alignment,sizeof(f32) * n);
    p->fy = aligned_alloc(alignment,sizeof(f32) * n);
    p->fz = aligned_alloc(alignment,sizeof(f32) * n);
    
    //
    init(p, n);
    
    const u64 s = sizeof(particle_t) * n;
    
    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
    
    //
    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
    
    //
    for (u64 i = 0; i < steps; i++)
    {
        
        omp_set_num_threads(tn);
        
        //Measure
        const f64 start = omp_get_wtime();
        
        move_particles(p, dt, n);
        
        const f64 end = omp_get_wtime();
        
        //Number of interactions/iteration
        const f32 h1 = (f32)(n) * (f32)(n);
        
        //Number of GFLOPs
        //Innermost loop (Newton's law)   : 36 FLOPs x n (innermost trip count) x n (outermost trip count)
        //Velocity update (outermost body):  6 FLOPs x n (outermost trip count)
        //Positions update                :  6 FLOPs x n
        const f32 h2 = (36.0 * h1 + 6.0 * (f32)n + 6.0 * (f32)n) * 1e-9;
        
        //Do not take warm up runs into account
        if (i >= warmup)
        {
            rate += h2 / (f32)(end - start);
            drate += (h2 * h2) / (f32)((end - start) * (end - start));
        }
        
        //
        printf("%5llu %10.3e %10.3e %8.1f %s\n",
               i,
               (end - start),
               h1 / (end - start),
               h2 / (end - start),
               (i < warmup) ? "(warm up)" : "");
        
        fflush(stdout);
    }
    
    //Average GFLOPs/s
    rate /= (f64)(steps - warmup);
    
    //Deviation in GFLOPs/s
    drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));
    
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
           "Average performance:", "", rate, drate);
    printf("-----------------------------------------------------\n");
    
    //
    free(p->x);
    free(p->y);
    free(p->z);
    free(p->vx);
    free(p->vy);
    free(p->vz);
    free(p->fx);
    free(p->fy);
    free(p->fz);
    free(p);
    //
    return 0;
}
