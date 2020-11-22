#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

struct Function_t {
    float (*func)(float);
};

void func_UseSrand();
float func_Sigmoid(float value);
float func_DSigmoid(float value);
float func_Uniform(float left, float right);
float func_SquaredError(float a, float b);
float func_CrossEntropy(float a, float b);
struct Function_t *func_GetActivationFunctions();
struct Function_t *func_GetDActivationFunctions();
int32_t func_TotalFunctions();
long func_Time();
void func_FreePointer(void *buffer);
int32_t func_SelectFromProbabilities(float *buffer, int32_t size);
float *func_NormalizeArray(float *buffer, int32_t size);
int32_t func_ArraySum(int32_t *buffer, int32_t size);
float func_RandomNumber(float min, float max);
float func_Error(float a, float b);