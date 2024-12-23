#include <math.h>

// 3x3 Matrix Operations
void multMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            result[i*3 + j] = a[i*3]*b[j] + a[i*3+1]*b[j+3] + a[i*3+2]*b[j+6];
}

void multMatVec3f(const double* m, const double* v, double* result) {
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    result[1] = m[3]*v[0] + m[4]*v[1] + m[5]*v[2];
    result[2] = m[6]*v[0] + m[7]*v[1] + m[8]*v[2];
}

void vecToDiagMat3f(const double* v, double* result) {
    for(int i = 0; i < 9; i++) result[i] = 0;
    result[0] = v[0];
    result[4] = v[1];
    result[8] = v[2];
}

void invMat3f(const double* m, double* result) {
    double det = m[0]*(m[4]*m[8] - m[7]*m[5]) - 
                m[1]*(m[3]*m[8] - m[5]*m[6]) + 
                m[2]*(m[3]*m[7] - m[4]*m[6]);
    
    if (det == 0.0f) {
        // Handle error case
        return;
    }
    
    double invDet = 1.0f/det;
    result[0] = invDet*(m[4]*m[8] - m[7]*m[5]);
    result[1] = invDet*(m[2]*m[7] - m[1]*m[8]);
    result[2] = invDet*(m[1]*m[5] - m[2]*m[4]);
    result[3] = invDet*(m[5]*m[6] - m[3]*m[8]);
    result[4] = invDet*(m[0]*m[8] - m[2]*m[6]);
    result[5] = invDet*(m[3]*m[2] - m[0]*m[5]);
    result[6] = invDet*(m[3]*m[7] - m[6]*m[4]);
    result[7] = invDet*(m[6]*m[1] - m[0]*m[7]);
    result[8] = invDet*(m[0]*m[4] - m[3]*m[1]);
}

void transpMat3f(const double* m, double* result) {
    result[0] = m[0]; result[1] = m[3]; result[2] = m[6];
    result[3] = m[1]; result[4] = m[4]; result[5] = m[7];
    result[6] = m[2]; result[7] = m[5]; result[8] = m[8];
}

void identMat3f(double* result) {
    for(int i = 0; i < 9; i++) result[i] = 0;
    result[0] = result[4] = result[8] = 1.0f;
}

void rotMat3f(char axis, double rads, double* result) {
    double s = sinf(rads), c = cosf(rads);
    switch(axis) {
        case 'x':
            result[0]=1; result[1]=0; result[2]=0;
            result[3]=0; result[4]=c; result[5]=-s;
            result[6]=0; result[7]=s; result[8]=c;
            break;
        case 'y':
            result[0]=c; result[1]=0; result[2]=s;
            result[3]=0; result[4]=1; result[5]=0;
            result[6]=-s; result[7]=0; result[8]=c;
            break;
        case 'z':
            result[0]=c; result[1]=-s; result[2]=0;
            result[3]=s; result[4]=c; result[5]=0;
            result[6]=0; result[7]=0; result[8]=1;
            break;
    }
}

void so3hat(const double* v, double* result) {
    result[0]=0; result[1]=-v[2]; result[2]=v[1];
    result[3]=v[2]; result[4]=0; result[5]=-v[0];
    result[6]=-v[1]; result[7]=v[0]; result[8]=0;
}

void so3vee(const double* m, double* result) {
    result[0] = m[7];
    result[1] = m[2];
    result[2] = m[3];
}

// Matrix arithmetic
void addMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 9; i++) result[i] = a[i] + b[i];
}

void subMat3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 9; i++) result[i] = a[i] - b[i];
}

void multScalMat3f(double s, const double* m, double* result) {
    for(int i = 0; i < 9; i++) result[i] = s * m[i];
}

// 4x4 Matrix Operations
void multMat4f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            result[i*4 + j] = a[i*4]*b[j] + a[i*4+1]*b[j+4] + 
                             a[i*4+2]*b[j+8] + a[i*4+3]*b[j+12];
}

// Vector Operations
void crossVec3f(const double* a, const double* b, double* result) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

void multScalVec3f(double s, const double* v, double* result) {
    for(int i = 0; i < 3; i++) result[i] = s * v[i];
}

void addVec3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++) result[i] = a[i] + b[i];
}

void subVec3f(const double* a, const double* b, double* result) {
    for(int i = 0; i < 3; i++) result[i] = a[i] - b[i];
}

double dotVec3f(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void normVec3f(const double* v, double* result) {
    double mag = sqrtf(dotVec3f(v, v));
    for(int i = 0; i < 3; i++) result[i] = v[i]/mag;
}

void inv4Mat4f(const double* m, double* result) {
    double s0 = m[0]*m[5] - m[4]*m[1];
    double s1 = m[0]*m[6] - m[4]*m[2];
    double s2 = m[0]*m[7] - m[4]*m[3];
    double s3 = m[1]*m[6] - m[5]*m[2];
    double s4 = m[1]*m[7] - m[5]*m[3];
    double s5 = m[2]*m[7] - m[6]*m[3];

    double c5 = m[10]*m[15] - m[14]*m[11];
    double c4 = m[9]*m[15] - m[13]*m[11];
    double c3 = m[9]*m[14] - m[13]*m[10];
    double c2 = m[8]*m[15] - m[12]*m[11];
    double c1 = m[8]*m[14] - m[12]*m[10];
    double c0 = m[8]*m[13] - m[12]*m[9];

    double det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    
    if (det == 0.0f) {
        // Handle error case
        return;
    }

    double invdet = 1.0f/det;

    result[0] = (m[5]*c5 - m[6]*c4 + m[7]*c3)*invdet;
    result[1] = (-m[1]*c5 + m[2]*c4 - m[3]*c3)*invdet;
    result[2] = (m[13]*s5 - m[14]*s4 + m[15]*s3)*invdet;
    result[3] = (-m[9]*s5 + m[10]*s4 - m[11]*s3)*invdet;

    result[4] = (-m[4]*c5 + m[6]*c2 - m[7]*c1)*invdet;
    result[5] = (m[0]*c5 - m[2]*c2 + m[3]*c1)*invdet;
    result[6] = (-m[12]*s5 + m[14]*s2 - m[15]*s1)*invdet;
    result[7] = (m[8]*s5 - m[10]*s2 + m[11]*s1)*invdet;

    result[8] = (m[4]*c4 - m[5]*c2 + m[7]*c0)*invdet;
    result[9] = (-m[0]*c4 + m[1]*c2 - m[3]*c0)*invdet;
    result[10] = (m[12]*s4 - m[13]*s2 + m[15]*s0)*invdet;
    result[11] = (-m[8]*s4 + m[9]*s2 - m[11]*s0)*invdet;

    result[12] = (-m[4]*c3 + m[5]*c1 - m[6]*c0)*invdet;
    result[13] = (m[0]*c3 - m[1]*c1 + m[2]*c0)*invdet;
    result[14] = (-m[12]*s3 + m[13]*s1 - m[14]*s0)*invdet;
    result[15] = (m[8]*s3 - m[9]*s1 + m[10]*s0)*invdet;
}

void multMatVec4f(const double* m, const double* v, double* result) {
    result[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3];
    result[1] = m[4]*v[0] + m[5]*v[1] + m[6]*v[2] + m[7]*v[3];
    result[2] = m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3];
    result[3] = m[12]*v[0] + m[13]*v[1] + m[14]*v[2] + m[15]*v[3];
}

// Helper functions for rotation matrices
void xRotMat3f(double rads, double* result) {
    rotMat3f('x', rads, result);
}

void yRotMat3f(double rads, double* result) {
    rotMat3f('y', rads, result);
}

void zRotMat3f(double rads, double* result) {
    rotMat3f('z', rads, result);
}

// Constants
#define K_F 0.0004905f
#define K_M 0.00004905f
#define L 0.25f
#define L_SQRT2 (L / sqrtf(2.0f))
#define G 9.81f
#define M 0.5f
#define DT 0.01f
#define OMEGA_MIN 30.0f
#define OMEGA_MAX 70.0f
#define OMEGA_STABLE 50.0f

// State variables
double omega[4];
double angular_velocity_B[3];
double linear_velocity_W[3];
double linear_position_W[3];
double R_W_B[3][3];  // 3x3 rotation matrix
double I[3] = {0.01f, 0.02f, 0.01f};

void init_drone_state(void) {
    // Initialize omegas
    for(int i = 0; i < 4; i++) {
        omega[i] = OMEGA_STABLE;
    }
    
    // Initialize angular velocity
    for(int i = 0; i < 3; i++) {
        angular_velocity_B[i] = 0.0f;
    }
    
    // Initialize linear velocity
    for(int i = 0; i < 3; i++) {
        linear_velocity_W[i] = 0.0f;
    }
    
    // Initialize position (0, 1, 0)
    linear_position_W[0] = 0.0f;
    linear_position_W[1] = 1.0f;
    linear_position_W[2] = 0.0f;
    
    // Initialize rotation matrix (identity matrix from rotation of 0 around all axes)
    double temp[3][3];
    double result[3][3];
    
    // Get rotation matrices for 0 rotation around each axis and multiply them
    xRotMat3f(0.0f, temp);
    yRotMat3f(0.0f, result);
    multMat3f(temp, result, R_W_B);
    zRotMat3f(0.0f, temp);
    multMat3f(R_W_B, temp, R_W_B);
}