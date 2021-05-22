//
// Created by quentin pierson on 09/05/2021.
//

typedef struct {
    float* values;
    int size;
} Model;

typedef struct {
    float** values;
    int x;
    int* y;
} Model2;

typedef struct {
    float*** values;
    int x;
    int* y;
} Model3;