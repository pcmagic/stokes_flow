#!/bin/csh -fe 

comm_list=("mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 25.595479 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c25.5955_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 10.000000 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c10.0000_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 3393.221772 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c3393.2218_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 1930.697729 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c1930.6977_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 37.275937 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c37.2759_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 115.139540 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c115.1395_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 22229.964825 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c22229.9648_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 625.055193 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c625.0552_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 56898.660290 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c56898.6603_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 12.067926 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c12.0679_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 65.512856 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c65.5129_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 4094.915062 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c4094.9151_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 517.947468 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c517.9475_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 12648.552169 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c12648.5522_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 54.286754 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c54.2868_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 95.409548 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c95.4095_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 2329.951811 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c2329.9518_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 202.358965 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c202.3590_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 21.209509 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c21.2095_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 1599.858720 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c1599.8587_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 754.312006 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c754.3120_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 244.205309 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c244.2053_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 44.984327 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c44.9843_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 10481.131342 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c10481.1313_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 910.298178 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c910.2982_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 1325.711366 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c1325.7114_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 14.563485 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c14.5635_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 2811.768698 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c2811.7687_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 1098.541142 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c1098.5411_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 429.193426 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c429.1934_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 4941.713361 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c4941.7134_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 167.683294 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c167.6833_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 68664.884500 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c68664.8845_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 17.575106 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c17.5751_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 82864.277285 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c82864.2773_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 79.060432 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c79.0604_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 294.705170 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c294.7052_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 138.949549 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c138.9495_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 15264.179672 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c15264.1797_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 26826.957953 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c26826.9580_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 8685.113738 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c8685.1137_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 18420.699693 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c18420.6997_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 100000.000000 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c100000.0000_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 47148.663635 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c47148.6636_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 32374.575428 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c32374.5754_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 5963.623317 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c5963.6233_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 30.888436 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c30.8884_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 39069.399371 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c39069.3994_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 7196.856730 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c7196.8567_th2.3562" "mpirun -np 24 python ../../obj_helicoid_dumb.py  -main_resistanceMatrix_dumb_sphere 1  -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rs 1.000000 -ds 0.200000 -dumb_d 5.000000 -dumb_theta 2.356194  -helicoid_r 355.648031 -helicoid_ndsk_each 4 -center_sphere_rs 1.000000 -center_sphere_ds 0.200000 -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -f c355.6480_th2.3562" ) 

txt_list=("c25.5955_th2.3562" "c10.0000_th2.3562" "c3393.2218_th2.3562" "c1930.6977_th2.3562" "c37.2759_th2.3562" "c115.1395_th2.3562" "c22229.9648_th2.3562" "c625.0552_th2.3562" "c56898.6603_th2.3562" "c12.0679_th2.3562" "c65.5129_th2.3562" "c4094.9151_th2.3562" "c517.9475_th2.3562" "c12648.5522_th2.3562" "c54.2868_th2.3562" "c95.4095_th2.3562" "c2329.9518_th2.3562" "c202.3590_th2.3562" "c21.2095_th2.3562" "c1599.8587_th2.3562" "c754.3120_th2.3562" "c244.2053_th2.3562" "c44.9843_th2.3562" "c10481.1313_th2.3562" "c910.2982_th2.3562" "c1325.7114_th2.3562" "c14.5635_th2.3562" "c2811.7687_th2.3562" "c1098.5411_th2.3562" "c429.1934_th2.3562" "c4941.7134_th2.3562" "c167.6833_th2.3562" "c68664.8845_th2.3562" "c17.5751_th2.3562" "c82864.2773_th2.3562" "c79.0604_th2.3562" "c294.7052_th2.3562" "c138.9495_th2.3562" "c15264.1797_th2.3562" "c26826.9580_th2.3562" "c8685.1137_th2.3562" "c18420.6997_th2.3562" "c100000.0000_th2.3562" "c47148.6636_th2.3562" "c32374.5754_th2.3562" "c5963.6233_th2.3562" "c30.8884_th2.3562" "c39069.3994_th2.3562" "c7196.8567_th2.3562" "c355.6480_th2.3562" ) 

echo ${comm_list[$1]} '>' ${txt_list[$1]}.txt '2>' ${txt_list[$1]}.err 
echo 
if [ ${2:-false} = true ]; then 
    ${comm_list[$1]} > ${txt_list[$1]}.txt 2> ${txt_list[$1]}.err 
fi 
