#include <iostream>
#include <random>
#include <vector>
#include "cuComplex.h"
#include <algorithm>

// takes a sparse matrix stored as array and makes it csr (compressed sparse row)
// a: stores values of the non-zero values of the matrix
// ia: stores cumulative nuber of non-zero elements up to i'th row (excluding)
// ja: stores column index of each element in the a vector
// total_nnz: number of non zero values in the sparse matrix
void makeCSR(cuDoubleComplex *matrix, int rows, int cols, cuDoubleComplex *a, int *ia, int *ja, int *total_nnz, int m) {
    int nnz = 0;	            // number of non zero values
	int nnz_per_row = 0;        // number of non zero values per row
	ia[0] = 0;

	for(int i = 0; i < (rows*cols*m); i++) {
		if(cuCreal(matrix[i]) != 0 && cuCimag(matrix[i]) != 0) {
			a[nnz] = matrix[i];	// store non zero value
			ja[nnz] = i % cols;   // store corresponding col to value	
			nnz ++;
			nnz_per_row ++;		        // store number of non zero values per row
		}
		if(i%cols == (cols -1)) {	    // if we are at the end of a row
            // stores cumulative number of non-zero elements at row number + 1 (this is a CSR convention)
            // length of ia = number of rows + 1
			ia[i/cols + 1] = ia[i/cols] + nnz_per_row;
			nnz_per_row = 0;
		}/*
        if(i%(rows*cols) == (rows*cols-1)) { // if we are at the end of a matrix
            nnz = 0;
        }*/
	}
    *total_nnz += nnz;
}

// prints the arrays into which our sparse matrix was converted into
void printCSR(cuDoubleComplex *a, int *ia, int *ja, int rows, int *total_nnz, int m) {
    std::cout<< "Array A: [";
    for(int i = 0; i < *total_nnz; i++) {
		std::cout<< "(" << cuCreal(a[i]) << ", " << cuCimag(a[i]) <<")   ";
	}
    std::cout<< "]\n";

	std::cout<< "Array IA: [";
	for(int j = 0; j < (rows*m+1); j++) {
		std::cout<< ia[j] << ", ";
	}
    std::cout<< "]\n";

	std::cout<< "Array JA: [";
	for(int k = 0; k < *total_nnz; k++) {
		std::cout<< ja[k] << ", ";
	}
    std::cout<< "]\n";

    std::cout<< "NNZ:" <<*total_nnz<< "\n";
}

// generate random int from 1 to 100
int getRand() {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist100(1, 100); // distribution in range [1, 100]

    return dist100(rng);
}

// fill 1 Matrix with values (ranged 0 to 100), row wise
void fillMatrix(cuDoubleComplex* m, int size) {
    for (int i = 0; i < size; i++) {
        // 80% chance of zero value
        if (getRand() > 20) {
            m[i] = make_cuDoubleComplex(0, 0);
        }
        else {
            m[i] = make_cuDoubleComplex(getRand(), getRand());
        }
    }
}

// fill Matrices with complex values (ranged 0 to 100), row wise, with k zeros per row
void fillStructuredMatrix(cuDoubleComplex* m, int totalrows, int cols, int k) {
    // fill matrices with nonzero values
    for (int value = 0; value < totalrows*cols; value++) {
        m[value] = make_cuDoubleComplex(getRand(), getRand());
    }

    // create vector with cols values
    std::vector<int> a(cols, 0);
    for(int count = 0; count < cols; count++) {
        a.at(count) = count;
    }
    
    // fill each row with k zero values
    for (int i = 0; i < totalrows; i++) {
        /* For linux PC: */
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(a.begin(), a.end(),g);

        /* For my Windows PC: 
        std::random_shuffle(a.begin(), a.end());
        */

        // make k random values zero
        for (int j = 0; j < k; j++) {
            m[i*cols+a.at(j)] = make_cuDoubleComplex(0, 0);
        }
    }
}

void printMatrix(cuDoubleComplex* m, int size, int columns) {
    for (int i = 0; i < size; i++) {
        std::cout << "(" << cuCreal(m[i]) << ", " << cuCimag(m[i]) << ")   ";
        if (i % columns == (columns - 1)) {
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

// print numMtx amount of matrices
void printMatrices(cuDoubleComplex* m, int singleSize, int columns, int numMtx) {
    std::cout << "MATRICES\n";
    for (int i = 0; i < singleSize * numMtx; i++) {
        std::cout << "(" << cuCreal(m[i]) << ", " << cuCimag(m[i]) << ")   ";
        if (i % columns == (columns - 1)) {
            std::cout << "\n";
        }
        if (i % singleSize == singleSize - 1) {
            std::cout << "-----------------------------------------------------\n";
        }
    }
    std::cout << "\n";
}

// fill Vectors (we fill v_y) with complex values from 0 to d_B
void fillVector(cuDoubleComplex* v, int size) {
    for (int i = 0; i < size; i++) {
        v[i] = make_cuDoubleComplex(getRand(), getRand());
    }
}

// must provide which vector should be printed and how many rows the vector has
void printVector(cuDoubleComplex* v, int rows) {
    for (int i = 0; i < rows; i++) {
        std::cout << "(" << cuCreal(v[i]) << ", " << cuCimag(v[i]) << ")\n";
    }
    std::cout << "\n";
}

// print numV amount of vectors
void printVectors(cuDoubleComplex* v, int rows, int numV) {
    std::cout << "VECTORS\n";
    for (int i = 0; i < rows * numV; i++) {
        std::cout << "(" << cuCreal(v[i]) << ", " << cuCimag(v[i]) << ")\n";

        if (i % rows == rows - 1) {
            std::cout << "-------------\n";
        }
    }
    std::cout << "\n";
}
