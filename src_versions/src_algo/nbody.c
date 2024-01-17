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

  f32  x, y, z;
  f32 vx, vy, vz;
  
} particle_t;

//tree node
typedef struct Node {
  f32 cx, cy, cz; //central
  f32 size;
  particle_t *restrict p;
  struct Node *fils[8];
  f32 m; //mass total
  f32 cmx, cmy, cmz;
} Node;


//create a node
Node *createNode(f32 x, f32 y, f32 z, f32 size) {
  Node *node = malloc(sizeof(Node));
  node->cx = x;
  node->cy = y;
  node->cz = z;
  node->size = size;
  node->p = NULL;
  for (u64 i = 0; i < 8; i++) {
      node->fils[i] = NULL;
  }
  node->m = 0;
  node->cmx = 0;
  node->cmy = 0;
  node->cmz = 0;
  return node;
}

//add particles into a node
void addParticle(Node *node, particle_t *restrict p) {
  if (node->p == NULL) {
      node->p = p;
  } else {
      for (u64 i = 0; i < 8; i++) {
          f32 f_size = node->size / 2; 
	  f32 f_x = node->cx + ((i & 1) ? f_size : -f_size) / 2;
          f32 f_y = node->cy + ((i & 2) ? f_size : -f_size) / 2;
          f32 f_z = node->cz + ((i & 4) ? f_size : -f_size) / 2;
          if (node->fils[i] == NULL) {
              node->fils[i] = createNode(f_x, f_y, f_z, f_size);
          }
          addParticle(node->fils[i], p);
      }
  }
}

//calculate the cmx,cmy,cmz and m  of a node
void computeMass(Node *node) {
  if (node->p != NULL) {
      node->m = 1;
      node->cmx = node->p->x;
      node->cmy = node->p->y;
      node->cmz = node->p->z;
  } else {
      for (u64 i = 0; i < 8; i++) {
          if (node->fils[i] != NULL) {
              computeMass(node->fils[i]);
              node->m += node->fils[i]->m;
              node->cmx += node->fils[i]->m * node->fils[i]->cmx;
              node->cmy += node->fils[i]->m * node->fils[i]->cmy;
              node->cmz += node->fils[i]->m * node->fils[i]->cmz;
          }
      }
      node->cmx /= node->m;
      node->cmy /= node->m;
      node->cmz /= node->m;
  }
}

//F
void computeForce(Node *node, particle_t *restrict p, f32 theta, f32 *fx, f32 *fy, f32 *fz) {
    const f32 dx = node->cmx - p->x;
    const f32 dy = node->cmy - p->y;
    const f32 dz = node->cmz - p->z;
    const f32 softening = 1e-20;
    const f32 dist = 1/sqrt(dx * dx + dy * dy + dz * dz + softening);
    
    if (node->size * dist < theta) {
    const f32 dist_3 = node->m * (dist * dist * dist);
    *fx += dist_3 * dx;
    *fy += dist_3 * dy;
    *fz += dist_3 * dz;
    }else{
        for (u64 i = 0; i < 8; i++) {
            if (node->fils[i] != NULL) {
                computeForce(node->fils[i], p, theta, fx, fy, fz);
            }
        }
    }
}

Node *createTree(particle_t *restrict p, u64 n) {
    f32 minX =  p[0].x, maxX = p[0].x;
    f32 minY =  p[0].y, maxY = p[0].y;
    f32 minZ =  p[0].z, maxZ = p[0].z;
    for (u64 i = 1; i < n; i++) {
        if ( p[i].x < minX) minX = p[i].x;
        if ( p[i].x > maxX) maxX = p[i].x;
        if ( p[i].y < minY) minY = p[i].y;
        if ( p[i].y > maxY) maxY = p[i].y;
        if ( p[i].z < minZ) minZ = p[i].z;
        if ( p[i].z > maxZ) maxZ = p[i].z;
    }
    printf("loop 1\n");
    f32 size = fmax(maxX - minX, fmax(maxY - minY, maxZ - minZ));
    f32 x = (minX + maxX) / 2;
    f32 y = (minY + maxY) / 2;
    f32 z = (minZ + maxZ) / 2;
    printf("bf create node \n");
    Node *root = createNode(x, y, z, size);
    printf("afteer create node \n");
    for (int i = 0; i < n; i++) {
	 printf("ADDING ,,,,,,,,,, %d \n", i);
        addParticle(root, &p[i]);
    }
    printf("loop2\n");
    computeMass(root);
    return root;
}



//
void init(particle_t *restrict p, u64 n)
{
  for (u64 i = 0; i < n; i++)
    {
      //
      u64 r1 = (u64)rand();
      u64 r2 = (u64)rand();
      f32 sign = (r1 > r2) ? 1 : -1;
      
      p[i].x = sign * (f32)rand() / (f32)RAND_MAX;
      p[i].y = (f32)rand() / (f32)RAND_MAX;
      p[i].z = sign * (f32)rand() / (f32)RAND_MAX;
      //
      p[i].vx = (f32)rand() / (f32)RAND_MAX;
      p[i].vy = sign * (f32)rand() / (f32)RAND_MAX;
      p[i].vz = (f32)rand() / (f32)RAND_MAX;

    }
}

//
void move_particles(particle_t *restrict p, Node *root, const f32 dt, u64 n)
{
    f32 theta = 0.5;
    //Newton's law: 17 FLOPs (Floating-Point Operations) per iteration
    for (u64 i = 0; i < n; i++)	
    { 
      f32 fx = 0.0;
      f32 fy = 0.0;
      f32 fz = 0.0;
      computeForce(root, &p[i], theta, &fx, &fy, &fz);
      //Update particle velocities using the previously computed net force: 6 FLOPs 
      p[i].vx += dt * fx; //19 (mul, add)
      p[i].vy += dt * fy; //21 (mul, add)
      p[i].vz += dt * fz; //23 (mul, add)
     }

  //Update positions: 6 FLOPs
  for (u64 i = 0; i < n; i++)
    {
      p[i].x += dt * p[i].vx;
      p[i].y += dt * p[i].vy;
      p[i].z += dt * p[i].vz;
      printf("%f %f %f\n", p[i].x, p[i].y, p[i].z);
    }
}

//
int main(int argc, char **argv)
{
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
  particle_t *restrict p = aligned_alloc(alignment,sizeof(particle_t) * n);
  //
  init(p, n);

  printf("not done\n");
  Node *root = createTree(p,n);
  printf("done \n");

  const u64 s = sizeof(particle_t) * n;
  
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
  
  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  
  //
  for (u64 i = 0; i < steps; i++)
    {
      //Measure
      const f64 start = omp_get_wtime();

      move_particles(p, root, dt, n);

      const f64 end = omp_get_wtime();

      //Number of interactions/iteration
      const f32 h1 = (f32)(n) * (f32)(n);

      //Number of GFLOPs
      //Innermost loop (Newton's law)   : 17 FLOPs x n (innermost trip count) x n (outermost trip count)
      //Velocity update (outermost body):  6 FLOPs x n (outermost trip count)
      //Positions update                :  6 FLOPs x n 
      const f32 h2 = (17.0 * h1 + 6.0 * (f32)n + 6.0 * (f32)n) * 1e-9;

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
  free(p);
  free(root);
  //
  return 0;
}
