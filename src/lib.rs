//! # svdlibrs
//!
//! A Rust port of LAS2 from SVDLIBC.
//!
//! It performs [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) on a sparse input [CscMatrix](https://docs.rs/nalgebra-sparse/0.5.0/nalgebra_sparse/csc/struct.CscMatrix.html) using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm), and returns the decomposition as [ndarray](https://docs.rs/ndarray/0.15.3/ndarray/) components.
//!
//! # SVD Examples
//!
//! ### SVD using [R](https://www.r-project.org/)
//!
//! ```text
//! $ Rscript -e 'options(digits=12);m<-matrix(1:9,nrow=3)^2;print(m);r<-svd(m);print(r);r$u%*%diag(r$d)%*%t(r$v)'
//!
//! • The input matrix: M
//!      [,1] [,2] [,3]
//! [1,]    1   16   49
//! [2,]    4   25   64
//! [3,]    9   36   81
//!
//! • The diagonal matrix (singular values): S
//! $d
//! [1] 123.676578742544   6.084527896514   0.287038004183
//!
//! • The left singular vectors: U
//! $u
//!                 [,1]            [,2]            [,3]
//! [1,] -0.415206840886 -0.753443585619 -0.509829424976
//! [2,] -0.556377565194 -0.233080213641  0.797569820742
//! [3,] -0.719755016815  0.614814099788 -0.322422608499
//!
//! • The right singular vectors: V
//! $v
//!                  [,1]            [,2]            [,3]
//! [1,] -0.0737286909592  0.632351847728 -0.771164846712
//! [2,] -0.3756889918995  0.698691000150  0.608842071210
//! [3,] -0.9238083467338 -0.334607272761 -0.186054055373
//!
//! • Recreating the original input matrix: r$u %*% diag(r$d) %*% t(r$v)
//!      [,1] [,2] [,3]
//! [1,]    1   16   49
//! [2,]    4   25   64
//! [3,]    9   36   81
//! ```
//!
//! ### SVD using svdlibrs
//!
//! ```rust
//! # extern crate ndarray;
//! # use ndarray::prelude::*;
//! use svdlibrs::{svdLAS2, SvdRec};
//!
//! let mut coo = nalgebra_sparse::coo::CooMatrix::<f64>::new(3, 3);
//! coo.push(0, 0, 1.0); coo.push(0, 1, 16.0); coo.push(0, 2, 49.0);
//! coo.push(1, 0, 4.0); coo.push(1, 1, 25.0); coo.push(1, 2, 64.0);
//! coo.push(2, 0, 9.0); coo.push(2, 1, 36.0); coo.push(2, 2, 81.0);
//!
//! let csc = nalgebra_sparse::csc::CscMatrix::from(&coo);
//! let svd: SvdRec = svdLAS2(
//!     &csc,                 // SVDLIBC (SMat) Matrix
//!     0,                    // upper limit of desired number of singular triplets (0 == all)
//!     &[-1.0e-30, 1.0e-30], // left,right end of interval containing unwanted eigenvalues
//!     1e-6,                 // relative accuracy of ritz values acceptable as eigenvalues
//!     3141,                 // a supplied random seed if > 0
//!     false,                // verbose output
//! )
//! .unwrap();
//! println!("svd.d = {}\n", svd.d);
//! println!("U =\n{:#?}\n", svd.ut.t());
//! println!("S =\n{:#?}\n", svd.s);
//! println!("V =\n{:#?}\n", svd.vt.t());
//!
//! // Note: svd.ut & svd.vt are returned in transposed form
//! // M = USV*
//! let M = svd.ut.t().dot(&Array2::from_diag(&svd.s)).dot(&svd.vt);
//!
//! let epsilon = 1.0e-12;
//! assert_eq!(svd.d, 3);
//!
//! assert!((M[[0, 0]] - 1.0).abs() < epsilon);
//! assert!((M[[0, 1]] - 16.0).abs() < epsilon);
//! assert!((M[[0, 2]] - 49.0).abs() < epsilon);
//! assert!((M[[1, 0]] - 4.0).abs() < epsilon);
//! assert!((M[[1, 1]] - 25.0).abs() < epsilon);
//! assert!((M[[1, 2]] - 64.0).abs() < epsilon);
//! assert!((M[[2, 0]] - 9.0).abs() < epsilon);
//! assert!((M[[2, 1]] - 36.0).abs() < epsilon);
//! assert!((M[[2, 2]] - 81.0).abs() < epsilon);
//!
//! assert!((svd.s[0] - 123.676578742544).abs() < epsilon);
//! assert!((svd.s[1] - 6.084527896514).abs() < epsilon);
//! assert!((svd.s[2] - 0.287038004183).abs() < epsilon);
//! ```
//!
//! # Output
//!
//! ```text
//! svd.d = 3
//!
//! U =
//! [[-0.4152068408862081, -0.7534435856189199, -0.5098294249756481],
//!  [-0.556377565193878, -0.23308021364108839, 0.7975698207417085],
//!  [-0.719755016814907, 0.6148140997884891, -0.3224226084985998]], shape=[3, 3], strides=[1, 3], layout=Ff (0xa), const ndim=2
//!
//! S =
//! [123.67657874254405, 6.084527896513759, 0.2870380041828973], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1
//!
//! V =
//! [[-0.07372869095916511, 0.6323518477280158, -0.7711648467120451],
//!  [-0.3756889918994792, 0.6986910001499903, 0.6088420712097343],
//!  [-0.9238083467337805, -0.33460727276072516, -0.18605405537270261]], shape=[3, 3], strides=[1, 3], layout=Ff (0xa), const ndim=2
//! ```

// ==================================================================================
// This is a functional port of "svdLAS2()" from Doug Rohde's SVDLIBC.
// It uses the same conceptual "workspace" storage as the C implementation.
// Most of the original function & variable names have been preserved.
// All C-style comments /* ... */ are from the original source, provided for context.
//
// dwf -- Wed May  5 16:48:01 MDT 2021
// ==================================================================================

/*
SVDLIBC License

The following BSD License applies to all SVDLIBC source code and documentation:

Copyright © 2002, University of Tennessee Research Foundation.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:


 Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 Neither the name of the University of Tennessee nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/***********************************************************************
 *                                                                     *
 *                        main()                                       *
 * Sparse SVD(A) via Eigensystem of A'A symmetric Matrix               *
 *                  (double precision)                                 *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  This sample program uses landr to compute singular triplets of A via
  the equivalent symmetric eigenvalue problem

  B x = lambda x, where x' = (u',v'), lambda = sigma**2,
  where sigma is a singular value of A,

  B = A'A , and A is m (nrow) by n (ncol) (nrow >> ncol),

  so that {u,sqrt(lambda),v} is a singular triplet of A.
  (A' = transpose of A)

  User supplied routines: svd_opa, opb, store, timer

  svd_opa(     x,y) takes an n-vector x and returns A*x in y.
  svd_opb(ncol,x,y) takes an n-vector x and returns B*x in y.

  Based on operation flag isw, store(n,isw,j,s) stores/retrieves
  to/from storage a vector of length n in s.

  User should edit timer() with an appropriate call to an intrinsic
  timing routine that returns elapsed user time.


  Local parameters
  ----------------

 (input)
  endl     left end of interval containing unwanted eigenvalues of B
  endr     right end of interval containing unwanted eigenvalues of B
  kappa    relative accuracy of ritz values acceptable as eigenvalues
             of B
         vectors is not equal to 1
  r        work array
  n        dimension of the eigenproblem for matrix B (ncol)
  dimensions   upper limit of desired number of singular triplets of A
  iterations   upper limit of desired number of Lanczos steps
  nnzero   number of nonzeros in A
  vectors  1 indicates both singular values and singular vectors are
         wanted and they can be found in output file lav2;
         0 indicates only singular values are wanted

 (output)
  ritz     array of ritz values
  bnd      array of error bounds
  d        array of singular values
  memory   total memory allocated in bytes to solve the B-eigenproblem


  Functions used
  --------------

  BLAS     svd_daxpy, svd_dscal, svd_ddot
  USER     svd_opa, svd_opb, timer
  MISC     write_header, check_parameters
  LAS2     landr


  Precision
  ---------

  All floating-point calculations are done in double precision;
  variables are declared as long and double.


  LAS2 development
  ----------------

  LAS2 is a C translation of the Fortran-77 LAS2 from the SVDPACK
  library written by Michael W. Berry, University of Tennessee,
  Dept. of Computer Science, 107 Ayres Hall, Knoxville, TN, 37996-1301

  31 Jan 1992:  Date written

  Theresa H. Do
  University of Tennessee
  Dept. of Computer Science
  107 Ayres Hall
  Knoxville, TN, 37996-1301
  internet: tdo@cs.utk.edu

***********************************************************************/

use nalgebra_sparse::csc::CscMatrix;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use std::mem;
extern crate ndarray;
use ndarray::prelude::*;

pub mod error;
use error::SvdLibError;

// ====================
//        Public
// ====================

/// Singular Value Decomposition components
///
/// # Fields
/// - d:  dimensionality (rank)
/// - ut: transpose of left singular vectors, the vectors are the rows of `ut`
/// - s:  singular values (length `d`)
/// - vt: transpose of right singular vectors, the vectors are the rows of `vt`
pub struct SvdRec {
    pub d: usize,
    pub ut: Array2<f64>,
    pub s: Array1<f64>,
    pub vt: Array2<f64>,
}

#[allow(non_snake_case)]
#[track_caller]
/// Compute a singular value decomposition
///
/// # Parameters
/// - csc: compressed sparse column matrix
/// - dim: upper limit of desired number of singular triplets (0 == all)
/// - end: left,right end of interval containing unwanted eigenvalues
/// - kappa: relative accuracy of ritz values acceptable as eigenvalues
/// - random_seed: a supplied seed if > 0, otherwise an internal seed will be generated
/// - verbose: output diagnostics to stderr
/// # Returns
/// SvdRec struct containing the decomposition
pub fn svdLAS2(
    csc: &CscMatrix<f64>,
    dim: usize,
    end: &[f64; 2],
    kappa: f64,
    random_seed: u32,
    verbose: bool,
) -> Result<SvdRec, SvdLibError> {
    let iterations = csc.nrows().min(csc.ncols());
    let dimensions = match dim.min(iterations) {
        n if n > 0 => n,
        _ => iterations,
    };

    if dimensions < 2 {
        return Err(SvdLibError::Las2Error(format!(
            "svdLAS2: insufficient dimensions: {}",
            dimensions
        )));
    }

    // If the matrix is wide, the SVD is computed on its transpose for speed
    let transpose = csc.ncols() as f64 >= (csc.nrows() as f64 * 1.2);
    let tm: CscMatrix<f64>;
    let A = match transpose {
        true => {
            tm = csc.transpose();
            &tm
        }
        false => csc,
    };

    let mut wrk = WorkSpace::new(A.ncols(), iterations)?;
    let mut store = Store::new(A.ncols())?;

    // Actually run the lanczos thing
    let mut neig = 0;
    let steps = lanso(
        A,
        dimensions,
        iterations,
        end,
        &mut wrk,
        &mut neig,
        &mut store,
        random_seed,
    )?;

    // Compute the singular vectors of matrix A
    let kappa = kappa.abs().max(eps34());
    let mut R = ritvec(A, dimensions, kappa, &mut wrk, steps, neig, &mut store)?;
    if verbose {
        let format_len = 50;
        eprintln!("{}", format!("{:~^1$}", "svdLas2", format_len));
        eprintln!("NON ZERO                = {}", A.nnz());
        eprintln!("DIMENSIONS              = {}", dimensions);
        eprintln!("TRANSPOSED              = {}", transpose);
        eprintln!("NUMBER OF LANCZOS STEPS = {}", steps + 1);
        eprintln!("RITZ VALUES STABILIZED  = {}", neig);
        eprintln!("SINGULAR VALUES FOUND   = {}", R.d);
        eprintln!("SIGNIFICANT VALUES      = {}", R.nsig);
    }

    // This swaps and transposes the singular matrices if A was transposed.
    if transpose {
        mem::swap(&mut R.Ut, &mut R.Vt);
    }
    //println!("R.S.len() = {}, R.nsig = {}, R.d = {}, R.Ut.rows = {}, R.Vt.rows = {}", R.S.len(), R.nsig, R.d, R.Ut.rows, R.Vt.rows);

    Ok(SvdRec {
        // Dimensionality (number of Ut,Vt rows & length of S)
        // Column vectors need to be truncated to this dimensionality
        d: R.d,
        ut: Array::from_shape_vec((R.Ut.rows, R.Ut.cols), R.Ut.value)?,
        s: Array::from_shape_vec(R.S.len(), R.S)?,
        vt: Array::from_shape_vec((R.Vt.rows, R.Vt.cols), R.Vt.value)?,
    })
}

//================================================================
//         Everything below is the private implementation
//================================================================

// ====================
//        Private
// ====================

const MAXLL: usize = 2;

fn eps34() -> f64 {
    f64::EPSILON.powf(0.75) // f64::EPSILON.sqrt() * f64::EPSILON.sqrt().sqrt();
}

/***********************************************************************
 *                                                                     *
 *                     store()                                         *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  store() is a user-supplied function which, based on the input
  operation flag, stores to or retrieves from memory a vector.


  Arguments
  ---------

  (input)
  n       length of vector to be stored or retrieved
  isw     operation flag:
        isw = 1 request to store j-th Lanczos vector q(j)
        isw = 2 request to retrieve j-th Lanczos vector q(j)
        isw = 3 request to store q(j) for j = 0 or 1
        isw = 4 request to retrieve q(j) for j = 0 or 1
  s       contains the vector to be stored for a "store" request

  (output)
  s       contains the vector retrieved for a "retrieve" request

  Functions used
  --------------

  BLAS     svd_dcopy

***********************************************************************/
struct Store {
    n: usize,
    vecs: Vec<Vec<f64>>,
}
impl Store {
    fn new(n: usize) -> Result<Self, SvdLibError> {
        Ok(Self { n, vecs: vec![] })
    }
    fn storq(&mut self, idx: usize, v: &[f64]) {
        while idx + MAXLL >= self.vecs.len() {
            self.vecs.push(vec![0.0; self.n]);
        }
        //self.vecs[idx + MAXLL] = v.to_vec();
        //self.vecs[idx + MAXLL][..self.n].clone_from_slice(&v[..self.n]);
        self.vecs[idx + MAXLL].copy_from_slice(v);
    }
    fn storp(&mut self, idx: usize, v: &[f64]) {
        while idx >= self.vecs.len() {
            self.vecs.push(vec![0.0; self.n]);
        }
        //self.vecs[idx] = v.to_vec();
        //self.vecs[idx][..self.n].clone_from_slice(&v[..self.n]);
        self.vecs[idx].copy_from_slice(v);
    }
    fn retrq(&mut self, idx: usize) -> &[f64] {
        &self.vecs[idx + MAXLL]
    }
    fn retrp(&mut self, idx: usize) -> &[f64] {
        &self.vecs[idx]
    }
}

struct WorkSpace {
    w0: Vec<f64>,     // workspace 0
    w1: Vec<f64>,     // workspace 1
    w2: Vec<f64>,     // workspace 2
    w3: Vec<f64>,     // workspace 3
    w4: Vec<f64>,     // workspace 4
    w5: Vec<f64>,     // workspace 5
    alf: Vec<f64>,    // array to hold diagonal of the tridiagonal matrix T
    eta: Vec<f64>,    // orthogonality estimate of Lanczos vectors at step j
    oldeta: Vec<f64>, // orthogonality estimate of Lanczos vectors at step j-1
    bet: Vec<f64>,    // array to hold off-diagonal of T
    bnd: Vec<f64>,    // array to hold the error bounds
    ritz: Vec<f64>,   // array to hold the ritz values
}
impl WorkSpace {
    fn new(n: usize, m: usize) -> Result<Self, SvdLibError> {
        Ok(Self {
            w0: vec![0.0; n],
            w1: vec![0.0; n],
            w2: vec![0.0; n],
            w3: vec![0.0; n],
            w4: vec![0.0; n],
            w5: vec![0.0; n],
            alf: vec![0.0; m],
            eta: vec![0.0; m],
            oldeta: vec![0.0; m],
            bet: vec![0.0; 1 + m],
            ritz: vec![0.0; 1 + m],
            bnd: vec![f64::MAX; 1 + m],
        })
    }
}

/* Row-major dense matrix.  Rows are consecutive vectors. */
#[allow(non_snake_case)]
#[derive(Debug)]
struct DMat {
    //long rows;
    //long cols;
    //double **value; /* Accessed by [row][col]. Free value[0] and value to free.*/
    rows: usize,
    cols: usize,
    value: Vec<f64>,
}

#[allow(non_snake_case)]
#[derive(Debug)]
struct SVDRec {
    //int d;      /* Dimensionality (rank) */
    //DMat Ut;    /* Transpose of left singular vectors. (d by m)
    //               The vectors are the rows of Ut. */
    //double *S;  /* Array of singular values. (length d) */
    //DMat Vt;    /* Transpose of right singular vectors. (d by n)
    //               The vectors are the rows of Vt. */
    d: usize,
    nsig: usize,
    Ut: DMat,
    S: Vec<f64>,
    Vt: DMat,
}

// =================================================================

// compare two floats within epsilon
#[track_caller]
fn compare(computed: f64, expected: f64) -> bool {
    (expected - computed).abs() < f64::EPSILON
}

#[track_caller]
/* Function sorts array1 and array2 into increasing order for array1 */
fn insert_sort(n: usize, array1: &mut [f64], array2: &mut [f64]) {
    for i in 1..n {
        let t1 = array1[i];
        let t2 = array2[i];
        let mut j: i32 = i as i32 - 1;
        while j >= 0 && t1 < array1[j as usize] {
            array1[(j + 1) as usize] = array1[j as usize];
            array2[(j + 1) as usize] = array2[j as usize];
            j -= 1;
        }
        array1[(j + 1) as usize] = t1;
        array2[(j + 1) as usize] = t2;
    }
}

#[allow(non_snake_case)]
#[track_caller]
// takes an n-vector x and returns A*x in y
fn svd_opa(A: &CscMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(
        x.len(),
        A.ncols(),
        "svd_opa: x must be A.ncols() in length, x = {}, A.ncols = {}",
        x.len(),
        A.ncols()
    );
    assert_eq!(
        y.len(),
        A.nrows(),
        "svd_opa: y must be A.nrows() in length, y = {}, A.nrows = {}",
        y.len(),
        A.nrows()
    );
    let (col_offsets, row_indices, values) = A.csc_data();

    y.fill(0.0);
    for (i, xval) in x.iter().enumerate() {
        for j in col_offsets[i]..col_offsets[i + 1] {
            y[row_indices[j]] += values[j] * xval;
        }
    }
}

#[allow(non_snake_case)]
#[track_caller]
// takes an n-vector x and returns B*x in y
fn svd_opb(A: &CscMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(
        x.len(),
        A.ncols(),
        "svd_opb: x must be A.ncols() in length, x = {}, A.ncols = {}",
        x.len(),
        A.ncols()
    );
    assert_eq!(
        y.len(),
        A.ncols(),
        "svd_opb: y must be A.ncols() in length, y = {}, A.ncols = {}",
        y.len(),
        A.ncols()
    );
    let (col_offsets, row_indices, values) = A.csc_data();

    let mut atx = vec![0.0; A.nrows()];
    svd_opa(A, x, &mut atx);

    y.fill(0.0);
    for (i, yval) in y.iter_mut().enumerate() {
        for j in col_offsets[i]..col_offsets[i + 1] {
            *yval += values[j] * atx[row_indices[j]];
        }
    }
}

#[track_caller]
// constant times a vector plus a vector
fn svd_daxpy(da: f64, x: &[f64], y: &mut [f64]) {
    for (xval, yval) in x.iter().zip(y.iter_mut()) {
        *yval += da * xval
    }
}

#[track_caller]
// finds the index of element having max absolute value
fn svd_idamax(n: usize, x: &[f64]) -> usize {
    match n {
        1 => 0,
        _ => {
            let mut imax: usize = 0;
            let mut dmax = x[imax].abs();
            for (i, xval) in x.iter().enumerate().take(n).skip(1) {
                if xval.abs() > dmax {
                    dmax = xval.abs();
                    imax = i;
                }
            }
            imax
        }
    }
}

#[track_caller]
// returns |a| if b is positive; else fsign returns -|a|
fn svd_fsign(a: f64, b: f64) -> f64 {
    match a >= 0.0 && b >= 0.0 || a < 0.0 && b < 0.0 {
        true => a,
        false => -a,
    }
}

#[allow(clippy::many_single_char_names)]
#[track_caller]
// finds sqrt(a^2 + b^2) without overflow or destructive underflow
fn svd_pythag(a: f64, b: f64) -> f64 {
    match a.abs().max(b.abs()) {
        n if n > 0.0 => {
            let mut p = n;
            let mut r = (a.abs().min(b.abs()) / p).powi(2);
            let mut t = 4.0 + r;
            while !compare(t, 4.0) {
                let s = r / t;
                let u = 1.0 + 2.0 * s;
                p *= u;
                r *= (s / u).powi(2);
                t = 4.0 + r;
            }
            p
        }
        _ => 0.0,
    }
}

#[track_caller]
// dot product of two vectors
fn svd_ddot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[track_caller]
// norm (length) of a vector
fn svd_norm(x: &[f64]) -> f64 {
    svd_ddot(x, x).sqrt()
}

#[track_caller]
// scales an input vector 'x', by a constant, storing in 'y'
fn svd_datx(da: f64, x: &[f64], y: &mut [f64]) {
    for (xval, yval) in x.iter().zip(y.iter_mut()) {
        *yval = da * xval
    }
}

#[track_caller]
// scales an input vector 'x' by a constant, modifying 'x'
fn svd_dscal(da: f64, x: &mut [f64]) {
    for item in x.iter_mut() {
        *item *= da;
    }
}

// copies a vector x to a vector y (reversed direction)
#[track_caller]
fn svd_dcopy(n: usize, offset: usize, x: &[f64], y: &mut [f64]) {
    assert!(n > 0, "svd_dcopy: unexpected inputs!");

    let start = n - 1;
    for i in 0..n {
        y[offset + start - i] = x[offset + i];
    }
}

/***********************************************************************
 *                                                                     *
 *                              imtqlb()                               *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  imtqlb() is a translation of a Fortran version of the Algol
  procedure IMTQL1, Num. Math. 12, 377-383(1968) by Martin and
  Wilkinson, as modified in Num. Math. 15, 450(1970) by Dubrulle.
  Handbook for Auto. Comp., vol.II-Linear Algebra, 241-248(1971).
  See also B. T. Smith et al, Eispack Guide, Lecture Notes in
  Computer Science, Springer-Verlag, (1976).

  The function finds the eigenvalues of a symmetric tridiagonal
  matrix by the implicit QL method.


  Arguments
  ---------

  (input)
  n      order of the symmetric tridiagonal matrix
  d      contains the diagonal elements of the input matrix
  e      contains the subdiagonal elements of the input matrix in its
         last n-1 positions.  e[0] is arbitrary

  (output)
  d      contains the eigenvalues in ascending order.  if an error
           exit is made, the eigenvalues are correct and ordered for
           indices 0,1,...ierr, but may not be the smallest eigenvalues.
  e      has been destroyed.
***********************************************************************/
#[track_caller]
#[allow(clippy::many_single_char_names)]
fn imtqlb(n: usize, d: &mut [f64], e: &mut [f64], bnd: &mut [f64]) -> Result<(), SvdLibError> {
    if n == 1 {
        return Ok(());
    }

    bnd[0] = 1.0;
    let last = n - 1;
    for i in 1..=last {
        bnd[i] = 0.0;
        e[i - 1] = e[i];
    }
    e[last] = 0.0;

    let mut i: i32 = 0;

    for l in 0..=last {
        let mut iteration = 0;
        while iteration <= 30 {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = d[m].abs() + d[m + 1].abs();
                if compare(test, test + e[m].abs()) {
                    break; // convergence = true;
                }
                m += 1;
            }
            let mut p = d[l];
            let mut f = bnd[l];
            if m == l {
                // order the eigenvalues
                let mut exchange = true;
                if l != 0 {
                    i = l as i32;
                    while i >= 1 && exchange {
                        if p < d[(i - 1) as usize] {
                            d[i as usize] = d[(i - 1) as usize];
                            bnd[i as usize] = bnd[(i - 1) as usize];
                            i -= 1;
                        } else {
                            exchange = false;
                        }
                    }
                }
                if exchange {
                    i = 0;
                }
                d[i as usize] = p;
                bnd[i as usize] = f;
                iteration = 31;
            } else {
                if iteration == 30 {
                    return Err(SvdLibError::ImtqlbError(
                        "imtqlb no convergence to an eigenvalue after 30 iterations".to_string(),
                    ));
                }
                iteration += 1;
                // ........ form shift ........
                let mut g = (d[l + 1] - p) / (2.0 * e[l]);
                let mut r = svd_pythag(g, 1.0);
                g = d[m] - p + e[l] / (g + svd_fsign(r, g));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let mut underflow = false;
                i = m as i32 - 1;
                while !underflow && i >= (l as i32) {
                    f = s * e[i as usize];
                    let b = c * e[i as usize];
                    r = svd_pythag(f, g);
                    e[(i + 1) as usize] = r;
                    if compare(r, 0.0) {
                        underflow = true;
                    } else {
                        s = f / r;
                        c = g / r;
                        g = d[(i + 1) as usize] - p;
                        r = (d[i as usize] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[(i + 1) as usize] = g + p;
                        g = c * r - b;
                        f = bnd[(i + 1) as usize];
                        bnd[(i + 1) as usize] = s * bnd[i as usize] + c * f;
                        bnd[i as usize] = c * bnd[i as usize] - s * f;
                        i -= 1;
                    }
                }
                // ........ recover from underflow .........
                if underflow {
                    d[(i + 1) as usize] -= p;
                } else {
                    d[l] -= p;
                    e[l] = g;
                }
                e[m] = 0.0;
            }
        }
    }
    Ok(())
}

/***********************************************************************
 *                                                                     *
 *                         startv()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function delivers a starting vector in r and returns |r|; it returns
  zero if the range is spanned, and ierr is non-zero if no starting
  vector within range of operator can be found.

  Parameters
  ---------

  (input)
  n      dimension of the eigenproblem matrix B
  wptr   array of pointers that point to work space
  j      starting index for a Lanczos run
  eps    machine epsilon (relative precision)

  (output)
  wptr   array of pointers that point to work space that contains
         r[j], q[j], q[j-1], p[j], p[j-1]
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
fn startv(
    A: &CscMatrix<f64>,
    wrk: &mut WorkSpace,
    step: u32,
    store: &mut Store,
    random_seed: u32,
) -> Result<f64, SvdLibError> {
    // get initial vector; default is random
    let mut rnm2 = svd_ddot(&wrk.w0, &wrk.w0);
    for id in 0..3 {
        if id > 0 || step > 0 || compare(rnm2, 0.0) {
            if random_seed > 0 {
                let mut bytes = [0; 32];
                for (i, b) in random_seed.to_le_bytes().iter().enumerate() {
                    bytes[i] = *b;
                }
                let mut seeded_rng = StdRng::from_seed(bytes);
                wrk.w0.fill_with(|| seeded_rng.gen_range(-1.0..1.0));
            } else {
                wrk.w0.fill_with(|| thread_rng().gen_range(-1.0..1.0));
            }
        }
        wrk.w3.copy_from_slice(&wrk.w0);

        // apply operator to put r in range (essential if m singular)
        svd_opb(A, &wrk.w3, &mut wrk.w0);
        wrk.w3.copy_from_slice(&wrk.w0);
        rnm2 = svd_ddot(&wrk.w3, &wrk.w3);
        if rnm2 > 0.0 {
            break;
        }
    }

    if rnm2 <= 0.0 {
        return Err(SvdLibError::StartvError(format!(
            "rnm2 <= 0.0, rnm2 = {}",
            rnm2
        )));
    }

    if step > 0 {
        for i in 0..step as usize {
            let v = store.retrq(i);
            svd_daxpy(-svd_ddot(&wrk.w3, v), v, &mut wrk.w0);
        }

        // make sure q[step] is orthogonal to q[step-1]
        svd_daxpy(-svd_ddot(&wrk.w4, &wrk.w0), &wrk.w2, &mut wrk.w0);
        wrk.w3.copy_from_slice(&wrk.w0);

        rnm2 = match svd_ddot(&wrk.w3, &wrk.w3) {
            dot if dot <= f64::EPSILON * rnm2 => 0.0,
            dot => dot,
        }
    }
    Ok(rnm2.sqrt())
}

/***********************************************************************
 *                                                                     *
 *                         stpone()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function performs the first step of the Lanczos algorithm.  It also
  does a step of extended local re-orthogonalization.

  Arguments
  ---------

  (input)
  n      dimension of the eigenproblem for matrix B

  (output)
  ierr   error flag
  wptr   array of pointers that point to work space that contains
           wptr[0]             r[j]
           wptr[1]             q[j]
           wptr[2]             q[j-1]
           wptr[3]             p
           wptr[4]             p[j-1]
           wptr[6]             diagonal elements of matrix T
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
fn stpone(
    A: &CscMatrix<f64>,
    wrk: &mut WorkSpace,
    store: &mut Store,
    random_seed: u32,
) -> Result<(f64, f64), SvdLibError> {
    // get initial vector; default is random
    let mut rnm = startv(A, wrk, 0, store, random_seed)?;
    if compare(rnm, 0.0) {
        return Err(SvdLibError::StponeError("rnm == 0.0".to_string()));
    }

    // normalize starting vector
    svd_datx(rnm.recip(), &wrk.w0, &mut wrk.w1);
    svd_dscal(rnm.recip(), &mut wrk.w3);

    // take the first step
    svd_opb(A, &wrk.w3, &mut wrk.w0);
    wrk.alf[0] = svd_ddot(&wrk.w0, &wrk.w3);
    svd_daxpy(-wrk.alf[0], &wrk.w1, &mut wrk.w0);
    let t = svd_ddot(&wrk.w0, &wrk.w3);
    wrk.alf[0] += t;
    svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
    wrk.w4.copy_from_slice(&wrk.w0);
    rnm = svd_norm(&wrk.w4);
    let anorm = rnm + wrk.alf[0].abs();
    Ok((rnm, f64::EPSILON.sqrt() * anorm))
}

/***********************************************************************
 *                                                                     *
 *                      lanczos_step()                                 *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function embodies a single Lanczos step

  Arguments
  ---------

  (input)
  n        dimension of the eigenproblem for matrix B
  first    start of index through loop
  last     end of index through loop
  wptr     array of pointers pointing to work space
  alf      array to hold diagonal of the tridiagonal matrix T
  eta      orthogonality estimate of Lanczos vectors at step j
  oldeta   orthogonality estimate of Lanczos vectors at step j-1
  bet      array to hold off-diagonal of T
  ll       number of intitial Lanczos vectors in local orthog.
             (has value of 0, 1 or 2)
  enough   stop flag
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn lanczos_step(
    A: &CscMatrix<f64>,
    wrk: &mut WorkSpace,
    first: usize,
    last: usize,
    ll: &mut u32,
    enough: &mut bool,
    rnm: &mut f64,
    tol: &mut f64,
    store: &mut Store,
) -> Result<usize, SvdLibError> {
    let eps1 = f64::EPSILON * (A.ncols() as f64).sqrt();
    let mut j = first;

    while j < last {
        mem::swap(&mut wrk.w1, &mut wrk.w2);
        mem::swap(&mut wrk.w3, &mut wrk.w4);

        store.storq(j - 1, &wrk.w2);
        if j - 1 < MAXLL {
            store.storp(j - 1, &wrk.w4);
        }
        wrk.bet[j] = *rnm;

        // restart if invariant subspace is found
        if compare(*rnm, 0.0) {
            *rnm = startv(A, wrk, j as u32, store, 0)?;
            if compare(*rnm, 0.0) {
                *enough = true;
            }
        }

        if *enough {
            // added by Doug...
            // These lines fix a bug that occurs with low-rank matrices
            mem::swap(&mut wrk.w1, &mut wrk.w2);
            // ...added by Doug
            break;
        }

        // take a lanczos step
        svd_datx(rnm.recip(), &wrk.w0, &mut wrk.w1);
        svd_dscal(rnm.recip(), &mut wrk.w3);
        svd_opb(A, &wrk.w3, &mut wrk.w0);
        svd_daxpy(-*rnm, &wrk.w2, &mut wrk.w0);
        wrk.alf[j] = svd_ddot(&wrk.w0, &wrk.w3);
        svd_daxpy(-wrk.alf[j], &wrk.w1, &mut wrk.w0);

        // orthogonalize against initial lanczos vectors
        if j <= MAXLL && wrk.alf[j - 1].abs() > 4.0 * wrk.alf[j].abs() {
            *ll = j as u32;
        }
        for i in 0..((j as u32 - 1).min(*ll)) as usize {
            let v1 = store.retrp(i);
            let t = svd_ddot(v1, &wrk.w0);
            let v2 = store.retrq(i);
            svd_daxpy(-t, v2, &mut wrk.w0);
            wrk.eta[i] = eps1;
            wrk.oldeta[i] = eps1;
        }

        // extended local reorthogonalization
        let t = svd_ddot(&wrk.w0, &wrk.w4);
        svd_daxpy(-t, &wrk.w2, &mut wrk.w0);
        if wrk.bet[j] > 0.0 {
            wrk.bet[j] += t;
        }
        let t = svd_ddot(&wrk.w0, &wrk.w3);
        svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
        wrk.alf[j] += t;
        wrk.w4.copy_from_slice(&wrk.w0);
        *rnm = svd_norm(&wrk.w4);
        let anorm = wrk.bet[j] + wrk.alf[j].abs() + *rnm;
        *tol = f64::EPSILON.sqrt() * anorm;

        // update the orthogonality bounds
        ortbnd(wrk, j, *rnm, eps1);

        // restore the orthogonality state when needed
        purge(A.ncols(), *ll, wrk, j, rnm, *tol, store);
        if *rnm <= *tol {
            *rnm = 0.0;
        }
        j += 1;
    }
    Ok(j)
}

/***********************************************************************
 *                                                                     *
 *                              purge()                                *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function examines the state of orthogonality between the new Lanczos
  vector and the previous ones to decide whether re-orthogonalization
  should be performed


  Arguments
  ---------

  (input)
  n        dimension of the eigenproblem for matrix B
  ll       number of intitial Lanczos vectors in local orthog.
  r        residual vector to become next Lanczos vector
  q        current Lanczos vector
  ra       previous Lanczos vector
  qa       previous Lanczos vector
  wrk      temporary vector to hold the previous Lanczos vector
  eta      state of orthogonality between r and prev. Lanczos vectors
  oldeta   state of orthogonality between q and prev. Lanczos vectors
  j        current Lanczos step

  (output)
  r        residual vector orthogonalized against previous Lanczos
             vectors
  q        current Lanczos vector orthogonalized against previous ones
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
fn purge(
    n: usize,
    ll: u32,
    wrk: &mut WorkSpace,
    step: usize,
    rnm: &mut f64,
    tol: f64,
    store: &mut Store,
) {
    if step < 2 + ll as usize {
        return;
    }

    let reps = f64::EPSILON.sqrt();
    let eps1 = f64::EPSILON * (n as f64).sqrt();

    let k = svd_idamax((step as u32 - (ll + 1)) as usize, &wrk.eta) + ll as usize;
    if wrk.eta[k].abs() > reps {
        let reps1 = eps1 / reps;
        let mut iteration = 0;
        let mut flag = true;
        while iteration < 2 && flag {
            if *rnm > tol {
                // bring in a lanczos vector t and orthogonalize both r and q against it
                let mut tq = 0.0;
                let mut tr = 0.0;
                for i in (ll as usize)..step as usize {
                    let v = store.retrq(i);
                    let t = svd_ddot(v, &wrk.w3);
                    tq += t.abs();
                    svd_daxpy(-t, v, &mut wrk.w1);
                    let t = svd_ddot(v, &wrk.w4);
                    tr += t.abs();
                    svd_daxpy(-t, v, &mut wrk.w0);
                }
                wrk.w3.copy_from_slice(&wrk.w1);
                let t = svd_ddot(&wrk.w0, &wrk.w3);
                tr += t.abs();
                svd_daxpy(-t, &wrk.w1, &mut wrk.w0);
                wrk.w4.copy_from_slice(&wrk.w0);
                *rnm = svd_norm(&wrk.w4);
                if tq <= reps1 && tr <= *rnm * reps1 {
                    flag = false;
                }
            }
            iteration += 1;
        }
        for i in (ll as usize)..=step as usize {
            wrk.eta[i] = eps1;
            wrk.oldeta[i] = eps1;
        }
    }
}

/***********************************************************************
 *                                                                     *
 *                          ortbnd()                                   *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Funtion updates the eta recurrence

  Arguments
  ---------

  (input)
  alf      array to hold diagonal of the tridiagonal matrix T
  eta      orthogonality estimate of Lanczos vectors at step j
  oldeta   orthogonality estimate of Lanczos vectors at step j-1
  bet      array to hold off-diagonal of T
  n        dimension of the eigenproblem for matrix B
  j        dimension of T
  rnm      norm of the next residual vector
  eps1     roundoff estimate for dot product of two unit vectors

  (output)
  eta      orthogonality estimate of Lanczos vectors at step j+1
  oldeta   orthogonality estimate of Lanczos vectors at step j
***********************************************************************/
#[track_caller]
fn ortbnd(wrk: &mut WorkSpace, step: usize, rnm: f64, eps1: f64) {
    if step < 1 {
        return;
    }
    if !compare(rnm, 0.0) && step > 1 {
        wrk.oldeta[0] = (wrk.bet[1] * wrk.eta[1] + (wrk.alf[0] - wrk.alf[step]) * wrk.eta[0]
            - wrk.bet[step] * wrk.oldeta[0])
            / rnm
            + eps1;
        if step > 2 {
            for i in 1..=(step as i32 - 2) as usize {
                wrk.oldeta[i] = (wrk.bet[i + 1] * wrk.eta[i + 1]
                    + (wrk.alf[i] - wrk.alf[step]) * wrk.eta[i]
                    + wrk.bet[i] * wrk.eta[i - 1]
                    - wrk.bet[step] * wrk.oldeta[i])
                    / rnm
                    + eps1;
            }
        }
    }
    wrk.oldeta[step - 1] = eps1;
    mem::swap(&mut wrk.oldeta, &mut wrk.eta);
    wrk.eta[step] = eps1;
}

/***********************************************************************
 *                                                                     *
 *                      error_bound()                                  *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function massages error bounds for very close ritz values by placing
  a gap between them.  The error bounds are then refined to reflect
  this.


  Arguments
  ---------

  (input)
  endl     left end of interval containing unwanted eigenvalues
  endr     right end of interval containing unwanted eigenvalues
  ritz     array to store the ritz values
  bnd      array to store the error bounds
  enough   stop flag
***********************************************************************/
#[track_caller]
fn error_bound(
    enough: &mut bool,
    endl: f64,
    endr: f64,
    ritz: &mut [f64],
    bnd: &mut [f64],
    step: usize,
    tol: f64,
) -> usize {
    // massage error bounds for very close ritz values
    let mid = svd_idamax(step + 1, bnd) as i32;

    let mut i: i32 = (((step + 1) + (step - 1)) / 2) as i32;
    while i > mid + 1 {
        if (ritz[(i - 1) as usize] - ritz[i as usize]).abs() < eps34() * ritz[i as usize].abs()
            && bnd[i as usize] > tol
            && bnd[(i - 1) as usize] > tol
        {
            bnd[(i - 1) as usize] =
                (bnd[i as usize].powi(2) + bnd[(i - 1) as usize].powi(2)).sqrt();
            bnd[i as usize] = 0.0;
        }
        i -= 1;
    }

    let mut i: i32 = (((step + 1) - (step - 1)) / 2) as i32;
    while i < mid - 1 {
        if (ritz[(i + 1) as usize] - ritz[i as usize]).abs() < eps34() * ritz[i as usize].abs()
            && bnd[i as usize] > tol
            && bnd[(i + 1) as usize] > tol
        {
            bnd[(i + 1) as usize] =
                (bnd[i as usize].powi(2) + bnd[(i + 1) as usize].powi(2)).sqrt();
            bnd[i as usize] = 0.0;
        }
        i += 1;
    }

    // refine the error bounds
    let mut neig = 0;
    let mut gapl = ritz[step] - ritz[0];
    for i in 0..=step {
        let mut gap = gapl;
        if i < step {
            gapl = ritz[i + 1] - ritz[i];
        }
        gap = gap.min(gapl);
        if gap > bnd[i] {
            bnd[i] = bnd[i] * (bnd[i] / gap);
        }
        if bnd[i] <= 16.0 * f64::EPSILON * ritz[i].abs() {
            neig += 1;
            if !*enough {
                *enough = endl < ritz[i] && ritz[i] < endr;
            }
        }
    }
    neig
}

/***********************************************************************
 *                                                                     *
 *                              imtql2()                               *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  imtql2() is a translation of a Fortran version of the Algol
  procedure IMTQL2, Num. Math. 12, 377-383(1968) by Martin and
  Wilkinson, as modified in Num. Math. 15, 450(1970) by Dubrulle.
  Handbook for Auto. Comp., vol.II-Linear Algebra, 241-248(1971).
  See also B. T. Smith et al, Eispack Guide, Lecture Notes in
  Computer Science, Springer-Verlag, (1976).

  This function finds the eigenvalues and eigenvectors of a symmetric
  tridiagonal matrix by the implicit QL method.


  Arguments
  ---------

  (input)
  nm     row dimension of the symmetric tridiagonal matrix
  n      order of the matrix
  d      contains the diagonal elements of the input matrix
  e      contains the subdiagonal elements of the input matrix in its
           last n-1 positions.  e[0] is arbitrary
  z      contains the identity matrix

  (output)
  d      contains the eigenvalues in ascending order.  if an error
           exit is made, the eigenvalues are correct but unordered for
           for indices 0,1,...,ierr.
  e      has been destroyed.
  z      contains orthonormal eigenvectors of the symmetric
           tridiagonal (or full) matrix.  if an error exit is made,
           z contains the eigenvectors associated with the stored
         eigenvalues.
***********************************************************************/
#[track_caller]
#[allow(clippy::many_single_char_names)]
fn imtql2(
    nm: usize,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
) -> Result<(), SvdLibError> {
    if n == 1 {
        return Ok(());
    }

    let last = n - 1;

    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[last] = 0.0;

    let nnm = n * nm;
    for l in 0..n {
        let mut iteration = 0;

        // look for small sub-diagonal element
        while iteration <= 30 {
            let mut m = l;
            while m < n {
                if m == last {
                    break;
                }
                let test = d[m].abs() + d[m + 1].abs();
                if compare(test, test + e[m].abs()) {
                    break; // convergence = true;
                }
                m += 1;
            }
            if m == l {
                break;
            }

            // error -- no convergence to an eigenvalue after 30 iterations.
            if iteration == 30 {
                return Err(SvdLibError::Imtql2Error(
                    "imtql2 no convergence to an eigenvalue after 30 iterations".to_string(),
                ));
            }
            iteration += 1;

            // form shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = svd_pythag(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + svd_fsign(r, g));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut underflow = false;
            let mut i: i32 = m as i32 - 1;
            while !underflow && i >= (l as i32) {
                let mut f = s * e[i as usize];
                let b = c * e[i as usize];
                r = svd_pythag(f, g);
                e[(i + 1) as usize] = r;
                if compare(r, 0.0) {
                    underflow = true;
                } else {
                    s = f / r;
                    c = g / r;
                    g = d[(i + 1) as usize] - p;
                    r = (d[i as usize] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[(i + 1) as usize] = g + p;
                    g = c * r - b;

                    // form vector
                    for k in (0..nnm).step_by(n) {
                        let index = k + i as usize;
                        f = z[index + 1];
                        z[index + 1] = s * z[index] + c * f;
                        z[index] = c * z[index] - s * f;
                    }
                    i -= 1;
                }
            } /* end while (underflow != FALSE && i >= l) */
            /*........ recover from underflow .........*/
            if underflow {
                d[(i + 1) as usize] -= p;
            } else {
                d[l] -= p;
                e[l] = g;
            }
            e[m] = 0.0;
        }
    }

    // order the eigenvalues
    for l in 1..n {
        let i = l - 1;
        let mut k = i;
        let mut p = d[i];
        for (j, item) in d.iter().enumerate().take(n).skip(l) {
            if *item < p {
                k = j;
                p = *item;
            }
        }

        // ...and corresponding eigenvectors
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in (0..nnm).step_by(n) {
                z.swap(j + i, j + k);
            }
        }
    }

    Ok(())
}

#[track_caller]
#[allow(non_snake_case)]
fn rotateArray(a: &mut [f64], x: usize) {
    let n = a.len();
    let mut j = 0;
    let mut start = 0;
    let mut t1 = a[0];

    for _ in 0..n {
        j = match j >= x {
            true => j - x,
            false => j + n - x,
        };

        let t2 = a[j];
        a[j] = t1;

        if j == start {
            j += 1;
            start = j;
            t1 = a[j];
        } else {
            t1 = t2;
        }
    }
}

/***********************************************************************
 *                                                                     *
 *                        ritvec()                                     *
 *          Function computes the singular vectors of matrix A         *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  This function is invoked by landr() only if eigenvectors of the A'A
  eigenproblem are desired.  When called, ritvec() computes the
  singular vectors of A and writes the result to an unformatted file.


  Parameters
  ----------

  (input)
  nrow       number of rows of A
  steps      number of Lanczos iterations performed
  fp_out2    pointer to unformatted output file
  n          dimension of matrix A
  kappa      relative accuracy of ritz values acceptable as
               eigenvalues of A'A
  ritz       array of ritz values
  bnd        array of error bounds
  alf        array of diagonal elements of the tridiagonal matrix T
  bet        array of off-diagonal elements of T
  w1, w2     work space

  (output)
  xv1        array of eigenvectors of A'A (right singular vectors of A)
  ierr       error code
             0 for normal return from imtql2()
             k if convergence did not occur for k-th eigenvalue in
               imtql2()
  nsig       number of accepted ritz values based on kappa

  (local)
  s          work array which is initialized to the identity matrix
             of order (j + 1) upon calling imtql2().  After the call,
             s contains the orthonormal eigenvectors of the symmetric
             tridiagonal matrix T
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn ritvec(
    A: &CscMatrix<f64>,
    dimensions: usize,
    kappa: f64,
    wrk: &mut WorkSpace,
    steps: usize,
    neig: usize,
    store: &mut Store,
) -> Result<SVDRec, SvdLibError> {
    let js = steps + 1;
    let jsq = js * js;
    let mut s = vec![0.0; jsq as usize];

    // initialize s to an identity matrix
    for i in (0..jsq).step_by(js + 1) {
        s[i] = 1.0;
    }

    let mut Vt = DMat {
        rows: dimensions,
        cols: A.ncols(),
        value: vec![0.0; A.ncols() * dimensions],
    };

    svd_dcopy(js, 0, &wrk.alf, &mut Vt.value);
    svd_dcopy(steps, 1, &wrk.bet, &mut wrk.w5);

    // on return from imtql2(), `R.Vt.value` contains eigenvalues in
    // ascending order and `s` contains the corresponding eigenvectors
    imtql2(js as usize, js as usize, &mut Vt.value, &mut wrk.w5, &mut s)?;

    let mut nsig = 0;
    let mut x = 0;
    let mut id2 = jsq - js;
    for k in 0..js as usize {
        if wrk.bnd[k] <= kappa * wrk.ritz[k].abs() && k as i32 > (js as i32 - neig as i32 - 1) {
            x = match x {
                0 => dimensions - 1,
                _ => x - 1,
            };

            let offset = x * Vt.cols;
            Vt.value[offset..(offset + Vt.cols)].fill(0.0);
            let mut idx = id2 as i32;
            for i in 0..js as usize {
                if s[idx as usize] != 0.0 {
                    for (j, item) in store.retrq(i).iter().enumerate().take(Vt.cols as usize) {
                        Vt.value[j + offset] += s[idx as usize] * item;
                    }
                }
                idx -= js as i32;
            }
            nsig += 1;
        }
        id2 += 1;
    }

    // Rotate the singular vectors and values.
    // `x` is now the location of the highest singular value.
    if x > 0 {
        rotateArray(&mut Vt.value, x * Vt.cols);
    }

    let mut Ut = DMat {
        rows: dimensions,
        cols: A.nrows(),
        value: vec![0.0; A.nrows() * dimensions],
    };
    let mut S = vec![0.0; dimensions];
    let d = dimensions.min(nsig);

    let mut tmp_vec = vec![0.0; Vt.cols as usize];
    for (i, sval) in S.iter_mut().enumerate().take(d as usize) {
        let vt_offset = i * Vt.cols as usize;
        let ut_offset = i * Ut.cols as usize;

        let vt_vec = &Vt.value[vt_offset..(vt_offset + Vt.cols as usize)];
        let ut_vec = &mut Ut.value[ut_offset..(ut_offset + Ut.cols as usize)];

        // multiply by matrix B first
        svd_opb(A, vt_vec, &mut tmp_vec);
        let t = svd_ddot(vt_vec, &tmp_vec);

        // store the Singular Value at S[i]
        *sval = t.sqrt();

        svd_daxpy(-t, vt_vec, &mut tmp_vec);
        wrk.bnd[js as usize] = svd_norm(&tmp_vec) * sval.recip();

        // multiply by matrix A to get (scaled) left s-vector
        svd_opa(A, vt_vec, ut_vec);
        svd_dscal(sval.recip(), ut_vec);
    }

    Ok(SVDRec {
        // Dimensionality (rank)
        d,

        // Significant values
        nsig,

        // DMat Ut  Transpose of left singular vectors. (d by m)
        //          The vectors are the rows of Ut.
        Ut,

        // Array of singular values. (length d)
        S,

        // DMat Vt  Transpose of right singular vectors. (d by n)
        //          The vectors are the rows of Vt.
        Vt,
    })
}

/***********************************************************************
 *                                                                     *
 *                          lanso()                                    *
 *                                                                     *
 ***********************************************************************/
/***********************************************************************

  Description
  -----------

  Function determines when the restart of the Lanczos algorithm should
  occur and when it should terminate.

  Arguments
  ---------

  (input)
  n         dimension of the eigenproblem for matrix B
  iterations    upper limit of desired number of lanczos steps
  dimensions    upper limit of desired number of eigenpairs
  endl      left end of interval containing unwanted eigenvalues
  endr      right end of interval containing unwanted eigenvalues
  ritz      array to hold the ritz values
  bnd       array to hold the error bounds
  wptr      array of pointers that point to work space:
              wptr[0]-wptr[5]  six vectors of length n
              wptr[6] array to hold diagonal of the tridiagonal matrix T
              wptr[9] array to hold off-diagonal of T
              wptr[7] orthogonality estimate of Lanczos vectors at
                step j
              wptr[8] orthogonality estimate of Lanczos vectors at
                step j-1
  (output)
  j         number of Lanczos steps actually taken
  neig      number of ritz values stabilized
  ritz      array to hold the ritz values
  bnd       array to hold the error bounds
  ierr      (globally declared) error flag
            ierr = 8192 if stpone() fails to find a starting vector
            ierr = k if convergence did not occur for k-th eigenvalue
                   in imtqlb()
***********************************************************************/
#[track_caller]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn lanso(
    A: &CscMatrix<f64>,
    dim: usize,
    iterations: usize,
    end: &[f64; 2],
    wrk: &mut WorkSpace,
    neig: &mut usize,
    store: &mut Store,
    random_seed: u32,
) -> Result<usize, SvdLibError> {
    let endl = end[0];
    let endr = end[1];

    /* take the first step */
    let rnm_tol = stpone(A, wrk, store, random_seed)?;
    let mut rnm = rnm_tol.0;
    let mut tol = rnm_tol.1;

    let eps1 = f64::EPSILON * (A.ncols() as f64).sqrt();
    wrk.eta[0] = eps1;
    wrk.oldeta[0] = eps1;
    let mut ll = 0;
    let mut first = 1;
    let mut last = iterations.min(dim.max(8) + dim);
    let mut enough = false;
    let mut j = 0;
    let mut intro = 0;

    while !enough {
        if rnm <= tol {
            rnm = 0.0;
        }

        // the actual lanczos loop
        let steps = lanczos_step(
            A,
            wrk,
            first,
            last,
            &mut ll,
            &mut enough,
            &mut rnm,
            &mut tol,
            store,
        )?;
        j = match enough {
            true => steps - 1,
            false => last - 1,
        };

        first = j + 1;
        wrk.bet[first] = rnm;

        // analyze T
        let mut l = 0;
        for _ in 0..j {
            if l > j {
                break;
            }
            let mut k = j;
            for i in l..=j {
                k = i;
                if compare(wrk.bet[i + 1], 0.0) {
                    break;
                }
                k += 1;
            }
            if k > j {
                k = j;
            }

            // now k is at the end of an unreduced submatrix
            let sz = k as i32 - l as i32;
            svd_dcopy((sz + 1) as usize, l, &wrk.alf, &mut wrk.ritz);
            svd_dcopy(sz as usize, l + 1, &wrk.bet, &mut wrk.w5);

            imtqlb(
                (sz + 1) as usize,
                &mut wrk.ritz[l..],
                &mut wrk.w5[l..],
                &mut wrk.bnd[l..],
            )?;

            for m in l..=k {
                wrk.bnd[m] = rnm * wrk.bnd[m].abs();
            }
            l = k + 1;
        }

        // sort eigenvalues into increasing order
        insert_sort(j + 1, &mut wrk.ritz, &mut wrk.bnd);

        *neig = error_bound(&mut enough, endl, endr, &mut wrk.ritz, &mut wrk.bnd, j, tol);

        // should we stop?
        if *neig < dim {
            if *neig == 0 {
                last = first + 9;
                intro = first;
            } else {
                last = first + 3.max(1 + ((j - intro) * (dim - *neig)) / *neig) as usize;
            }
            last = last.min(iterations);
        } else {
            enough = true
        }
        enough = enough || first >= iterations;
    }
    store.storq(j, &wrk.w1);
    Ok(j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn basic_2x2() {
        // [
        //   [ 4,  0 ],
        //   [ 3, -5 ]
        // ]
        let mut coo = nalgebra_sparse::coo::CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 4.0);
        coo.push(1, 0, 3.0);
        coo.push(1, 1, -5.0);

        let csc = nalgebra_sparse::csc::CscMatrix::from(&coo);

        let svd: SvdRec = svdLAS2(
            &csc,                 // SVDLIBC "A" (SMat) Matrix
            0,                    // upper limit of desired number of singular triplets (0 == all)
            &[-1.0e-30, 1.0e-30], // left,right end of interval containing unwanted eigenvalues
            1e-6,                 // relative accuracy of ritz values acceptable as eigenvalues
            0,                    // a supplied random seed if > 0
            false,                // verbose output
        )
        .unwrap();
        println!("svd.d = {}", svd.d);
        println!("U = {:#?}", svd.ut.t());
        println!("S = {:#?}", svd.s);
        println!("V = {:#?}", svd.vt.t());

        // Note: svd.ut & svd.vt are returned in transposed form
        // M = USV*
        let M = svd.ut.t().dot(&Array2::from_diag(&svd.s)).dot(&svd.vt);

        let epsilon = 1.0e-12;
        assert_eq!(svd.d, 2);
        assert!((M[[0, 0]] - 4.0).abs() < epsilon);
        assert!((M[[0, 1]] - 0.0).abs() < epsilon);
        assert!((M[[1, 0]] - 3.0).abs() < epsilon);
        assert!((M[[1, 1]] - -5.0).abs() < epsilon);

        assert!((svd.s[0] - 6.3245553203368).abs() < epsilon);
        assert!((svd.s[1] - 3.1622776601684).abs() < epsilon);
    }
}
