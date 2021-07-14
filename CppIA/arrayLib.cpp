#include "header.h"
#include "fonctionnalite.h"

float* cut_float_array(float* array, int start, int end){
    int size = end-start;

    float* new_array = (float*) malloc(sizeof (float ) * size);

    for(int i = 0; i < size; i++){
        new_array[i] = array[start+i];
    }
    return new_array;
}

float* copy_array(float* array, int size){
    float* new_array = (float*) malloc(sizeof (float ) * size);

    for(int i = 0; i < size; i++){
        new_array[i] = array[i];
    }
    return new_array;
}

float* insert_array(float* array, int size,int pos, float value){
    int newSize = size+1;
    float* new_array = (float*) malloc(sizeof (float ) * (newSize));

    if(pos >= newSize){
        pos = size;
    }
    if(pos < 0){
        pos = 0;
    }

    int j = 0;
    for(int i = 0; i < newSize; i++, j++){
        if(i == pos){
            new_array[i] = value;
            j -= 1;
        }else{
            new_array[i] = array[j];
        }
    }

    free(array);

    return new_array;
}

void printArray(float* array, int size){
    std::cout << "[ ";

    for(int i = 0; i < size; i++){
        std::cout << array[i] << ", ";
    }

    std::cout << "] "<< "\n";
}

void printArray(Model* model){
    std::cout << "[ ";

    for(int i = 0; i < model->size; i++){
        std::cout << model->values[i] << ", ";
    }

    std::cout << "] "<< "\n";
}

void printW(Model3* W){
    std::cout << "[ ";
    for(int l=0; l< W->x ;l++){ //
        int imax= W->y[l-1]+1 ;
        std::cout << "[ ";
        if(l==0){
            std::cout << "], ";
            continue;
        }
        for (int i = 0; i < imax; i++){
            std::cout << "[ ";
            int jmax = W->y[l] + 1;
            for (int j = 0; j < jmax; j++){
                std::cout << W->values[l][i][j] << ", ";
            }
            std::cout << "], ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

void printX(Model2* X){
    std::cout << "[ ";
    for(int l=0; l< X->x ;l++){ //
        int iMax = X->y[l];
        std::cout << "[ ";

        for (int i = 0; i < iMax; i++){
            std::cout << X->values[l][i] << ", ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

Model* CreateModel(float* array, int size){
    Model* model = (Model*) (malloc(sizeof(Model)));
    model->size = size;
    model->values = copy_array(array, size);
    return model;
}

Model2* CreateModel2(int size){
    Model2* model2 = (Model2*) (malloc( sizeof(Model2)));
    model2->values = (float **) (malloc(sizeof(float **) * size));

    model2->x = size;
    model2->y = (int *) (malloc(sizeof(int) * size));

    return model2;
}

Model2* model2_set_zero(int x, int y){
    Model2* model = CreateModel2(x);

    for(int i = 0; i < x; i +=1){
        model->y[i] = y;
    }

    for(int i = 0; i < x; i+=1){
        model->values[i] = (float *) (malloc(sizeof(float *) * y));
        for(int j = 0; j < y; j+=1){
            model->values[i][j] = 0.0f;
        }
    }

    return model;
}

float* TakeLast(Model2* model){
    int xSize = model->x;
    float * tab = (float *) (malloc(sizeof(float *) * model->y[xSize-1] -1));

    for (int i = 1; i < model->y[xSize-1]; i++){
        tab[i-1] = model->values[xSize-1][i];
    }
    return tab;
}

void freeW(Model3* W) {
    for(int l=0; l < W->x ;l++){ //
        int imax= W->y[l-1]+1;

        if(l!=0) {
            for (int i = 0; i < imax; i++) {

                free(W->values[l][i]);

            }
        }
        free(W->values[l]);
    }

    free(W->values);
    free(W->y);
    free(W);
}

void freeX(Model2* X) {
    for(int l=0; l< X->x ;l++){ //
        free(X->values[l]);
    }

    free(X->values);
    free(X->y);
    free(X);
}

int minimumArray(float* array , int sizeArray){
    int min = INT_MAX;
    for (int i=0 ; i < sizeArray; i++){
        if(i < min) {
            min = i;
        }
    }
    return min;
}

int maximumArray(float * array,int sizeArray) {
    int max = INT_MIN;
    for (int i = 0; i < sizeArray; i++) {
        if (i > max) {
            max = i;
        }
    }
}