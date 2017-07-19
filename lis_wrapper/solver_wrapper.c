#include "solver_wrapper.h"


void solver_wrapper(const LIS_MATRIX *rs_m,         // velocity = rs_m * force
                    LIS_VECTOR *force,
                    const LIS_VECTOR *velocity,
                    char *solver_option,      // Options
                    LIS_INT *iter,                  // Number of iterations
                    LIS_REAL *time,                 // The total iteration time in seconds
                    LIS_REAL *resid)                // Relative residual
{
    double start=lis_wtime();
    LIS_SOLVER solver;
    LIS_INT	err=0, temp1=1, temp2=0;

    char* temp4 = "Null";
    char** temp3 = &temp4;
    lis_initialize( &temp1, &temp3 );

    lis_vector_duplicate( *rs_m, force );
    err = lis_solver_create( &solver );
//    CHKERR( err );
    lis_solver_set_option( solver_option, solver );
    err = lis_solve( *rs_m, *velocity, *force, solver );
//    CHKERR( err );
    lis_solver_get_iterex( solver, iter, &temp1, &temp2 );
    lis_solver_get_residualnorm( solver, resid );
    lis_solver_destroy( solver );
    lis_finalize();

    *time = lis_wtime() - start;
    return;
}

//void time_duration(LIS_REAL* duration,
//                   const struct timeval* begin_time,
//                   const struct timeval* final_time)
//{
//    LIS_INT tv_sec, tv_usec;
//
//    tv_sec = final_time->tv_sec - begin_time->tv_sec;
//    tv_usec = final_time->tv_usec - begin_time->tv_usec;
//
//    if ( tv_usec < 0 )
//    {
//        tv_sec--;
//        tv_usec + 1000000;
//    }
//
//    *duration = (LIS_REAL)tv_sec + (LIS_REAL)tv_usec/1000000.;
//
//    return;
//};
