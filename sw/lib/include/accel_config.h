/*
make each one if allready exits no need
*/
#ifndef ACCEL_CONFIG_H
#define ACCEL_CONFIG_H
#ifndef ACCEL_DIM
#define ACCEL_DIM 4
#endif

#if ACCEL_DIM < 1
#error "ACCEL_DIM must be at least 1"
#endif

#endif