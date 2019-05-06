#ifndef SOLVER_WRAPPER_H_INCLUDED
#define SOLVER_WRAPPER_H_INCLUDED

#include <sys/time.h>
#include <memory.h>
#include "lis.h"


struct MATRIX_STRUCT
{
    double** value;
    int
};


// wrapper of solver, Zhang Ji, 20160430
void solver_wrapper(const LIS_MATRIX *rs_m,         // velocity = rs_m * force
                    LIS_VECTOR *force,
                    const LIS_VECTOR *velocity,
                    char *solver_option,            // Options
                    LIS_INT *iter,                  // Number of iterations
                    LIS_REAL *time,                 // The total iteration time in seconds
                    LIS_REAL *resid);               // Relative residual


//// solver wrapper for python, Zhang Ji, 20160503
//void solver_python(const double** rs_m,         // matrix
//                   double* force,               // vector
//                   const double* velocity,
//                   int size)


//void time_duration(LIS_REAL* duration,
//                   const struct timeval* begin_time,
//                   const struct timeval* final_time);


#endif // SOLVER_WRAPPER_H_INCLUDED
