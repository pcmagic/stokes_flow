#ifdef HAVE_CONFIG_H
#include "lis_config.h"
#else
#ifdef HAVE_CONFIG_WIN_H
#include "lis_config_win.h"
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lis.h"
#include "solver_wrapper.h"

#undef __FUNC__
#define __FUNC__ "main"
LIS_INT main( LIS_INT argc, char* argv[] )
{
    LIS_INT nprocs, my_rank;
    int int_nprocs, int_my_rank;
    LIS_MATRIX rs_m;       // velocity = rs_m * force
    LIS_VECTOR force, velocity;
    LIS_INT	err, iter;
    LIS_REAL time, resid;
    LIS_DEBUG_FUNC_IN;

//    lis_initialize( &argc, &argv );

#ifdef USE_MPI
    MPI_Comm_size( MPI_COMM_WORLD, &int_nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &int_my_rank );
    nprocs = int_nprocs;
    my_rank = int_my_rank;
#else
    nprocs  = 1;
    my_rank = 0;
#endif

    if ( argc < 4 )
    {
        if ( my_rank == 0 )
        {
            printf( "Usage: %s matrix_filename velocity_filename force_filename\n", argv[0] );
        }
        CHKERR( 1 );
    }

    if ( my_rank == 0 )
    {
        printf( "\n" );
#ifdef _LONG__LONG
        printf( "number of processes = %lld\n", nprocs );
#else
        printf( "number of processes = %d\n", nprocs );
#endif
    }

#ifdef _OPENMP
    if ( my_rank == 0 )
    {
        printf( "max number of threads = %d\n", omp_get_num_procs() );
        printf( "number of threads = %d\n", omp_get_max_threads() );
    }
#endif

    /* create matrix and vectors */
    lis_matrix_create( LIS_COMM_WORLD, &rs_m );
    lis_input_matrix( rs_m, argv[1] );
    lis_vector_create( LIS_COMM_WORLD, &velocity );
    lis_input_vector( velocity, argv[2] );

    char *solver_option = "-i gmres -f quad";
    LIS_REAL total_time = 0.;
    LIS_INT loops = 10;
    for (int i0 = 0; i0 < loops; i0++)
    {
        solver_wrapper( &rs_m, &force, &velocity, solver_option, &iter, &time, &resid );
        total_time += time;
    }
    time = total_time / loops;

    if ( my_rank == 0 )
    {
        printf( "Solver option: %s\n", solver_option );
#ifdef _LONG__LONG
        printf( "number of iterations = %lld \n", iter );
#else
        printf( "number of iterations = %d \n", iter );
#endif
#ifdef _LONG__DOUBLE
        printf( "elapsed time         = %Le sec.\n", time );
        printf( "relative residual    = %Le\n\n", resid );
#else
        printf( "elapsed time         = %e sec.\n", time );
        printf( "relative residual    = %e\n\n", resid );
#endif
    }

//	lis_output_vector(force, LIS_FMT_MM, "force_out.txt");
//	lis_output_vector(velocity, LIS_FMT_MM, "velocity_out.txt");
//	lis_output_matrix(rs_m, LIS_FMT_MM, "rs_m_out.txt");

    lis_matrix_destroy( rs_m );
    lis_vector_destroy( velocity );
    lis_vector_destroy( force );
//    lis_finalize();

    LIS_DEBUG_FUNC_OUT;
    return 0;
}
