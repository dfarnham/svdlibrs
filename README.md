# svdlibrs &emsp; [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/badge/crates.io-v0.4.0-blue
[crates.io]: https://crates.io/crates/svdlibrs

A library that computes an svd on a sparse matrix, typically a large sparse matrix 

A Rust port of LAS2 from SVDLIBC

This is a functional port (mostly a translation) of the algorithm as seen in Doug Rohde's SVDLIBC

This library performs [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) on a sparse input [CscMatrix](https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/csc/struct.CscMatrix.html) using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) and returns the decomposition as [ndarray](https://docs.rs/ndarray/latest/ndarray/) components.

# Usage

Input: [CscMatrix](https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/csc/struct.CscMatrix.html)

Output: decomposition U,S,V where U,V are [Array2](https://docs.rs/ndarray/latest/ndarray/type.Array2.html) and S is [Array1](https://docs.rs/ndarray/latest/ndarray/type.Array1.html)

The above [ndarray](https://docs.rs/ndarray/latest/ndarray/) components along with the computed dimension and informational diagnostics are packaged as a Result

```rust
use svdlibrs::svd;

/// svd on a sparse matrix
let svd = svd(&csc)?;
```
```rust
use svdlibrs::svd_dim;

/// svd on a sparse matrix specifying a desired dimension, 3 in this example.
let svd = svd_dim(&csc, 3)?;
```
```rust
use svdlibrs::svd_dim_seed;

/// svd on a sparse matrix requesting the dimension
/// and supplying a fixed seed to the LAS2 algorithm
let svd = svd_dim_seed(&csc, dimension, 12345)?;
```
# The above examples use svdLAS2 as shown below
```rust
use svdlibrs::{svd, svd_dim, svd_dim_seed, svdLAS2, SvdRec};

/// These are equivalent:
let svd = svd(&csc)?;
let svd = svdLAS2(&csc, 0, 0, &[-1.0e-30, 1.0e-30], 1.0e-6, 0)?;

/// These are equivalent:
let svd = svd_dim(&csc, dimension)?;
let svd = svdLAS2(&csc, dimension, 0, &[-1.0e-30, 1.0e-30], 1.0e-6, 0)?;

/// These are equivalent:
let random_seed = 12345;
let svd = svd_dim_seed(&csc, dimension, random_seed)?;
let svd = svdLAS2(&csc, dimension, 0, &[-1.0e-30, 1.0e-30], 1.0e-6, random_seed)?;

/// Parameter description
let svd: SvdRec = svdLAS2(
    &csc,                 // sparse matrix
    0,                    // upper limit of desired number of singular triplets (0 = all)
    0,                    // number of algorithm iterations (0 = smaller of csc rows or columns),
                          // the upper limit of desired number of lanczos steps
    &[-1.0e-30, 1.0e-30], // left, right end of interval containing unwanted eigenvalues,
                          // typically small values centered around zero, e.g. [-1.0e-30, 1.0e-30]
    1.0e-6,               // relative accuracy of ritz values acceptable as eigenvalues
    0,                    // a supplied seed if > 0, otherwise an internal seed will be generated
)?;
```

# More SVD Examples

### SVD using [R](https://www.r-project.org/)

```text
$ Rscript -e 'options(digits=12);m<-matrix(1:9,nrow=3)^2;print(m);r<-svd(m);print(r);r$u%*%diag(r$d)%*%t(r$v)'

• The input matrix: M
     [,1] [,2] [,3]
[1,]    1   16   49
[2,]    4   25   64
[3,]    9   36   81

• The diagonal matrix (singular values): S
$d
[1] 123.676578742544   6.084527896514   0.287038004183

• The left singular vectors: U
$u
                [,1]            [,2]            [,3]
[1,] -0.415206840886 -0.753443585619 -0.509829424976
[2,] -0.556377565194 -0.233080213641  0.797569820742
[3,] -0.719755016815  0.614814099788 -0.322422608499

• The right singular vectors: V
$v
                 [,1]            [,2]            [,3]
[1,] -0.0737286909592  0.632351847728 -0.771164846712
[2,] -0.3756889918995  0.698691000150  0.608842071210
[3,] -0.9238083467338 -0.334607272761 -0.186054055373

• Recreating the original input matrix: r$u %*% diag(r$d) %*% t(r$v)
     [,1] [,2] [,3]
[1,]    1   16   49
[2,]    4   25   64
[3,]    9   36   81
```

### SVD using svdlibrs

• Cargo.toml dependencies
```text
[dependencies]
svdlibrs = "0.3.0"
nalgebra-sparse = "0.6.0"
ndarray = "0.15.4"
```

```rust
extern crate ndarray;
use ndarray::prelude::*;
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};
use svdlibrs::svd;

fn main() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 1.0); coo.push(0, 1, 16.0); coo.push(0, 2, 49.0);
    coo.push(1, 0, 4.0); coo.push(1, 1, 25.0); coo.push(1, 2, 64.0);
    coo.push(2, 0, 9.0); coo.push(2, 1, 36.0); coo.push(2, 2, 81.0);

    let csc = CscMatrix::from(&coo);
    let svd = svd_from_seed(&csc,0,3141).unwrap();
    assert_eq!(svd.d, svd.ut.nrows());
    assert_eq!(svd.d, svd.s.dim());
    assert_eq!(svd.d, svd.vt.nrows());
    println!("svd.d = {}\n", svd.d);
    println!("U =\n{:#?}\n", svd.ut.t());
    println!("S =\n{:#?}\n", svd.s);
    println!("V =\n{:#?}\n", svd.vt.t());

    // Note: svd.ut & svd.vt are returned in transposed form
    // M = USV*
    let matrix_approximation = svd.ut.t().dot(&Array2::from_diag(&svd.s)).dot(&svd.vt);

    let epsilon = 1.0e-12;
    assert_eq!(svd.d, 3);

    assert!((matrix_approximation[[0, 0]] - 1.0).abs() < epsilon);
    assert!((matrix_approximation[[0, 1]] - 16.0).abs() < epsilon);
    assert!((matrix_approximation[[0, 2]] - 49.0).abs() < epsilon);
    assert!((matrix_approximation[[1, 0]] - 4.0).abs() < epsilon);
    assert!((matrix_approximation[[1, 1]] - 25.0).abs() < epsilon);
    assert!((matrix_approximation[[1, 2]] - 64.0).abs() < epsilon);
    assert!((matrix_approximation[[2, 0]] - 9.0).abs() < epsilon);
    assert!((matrix_approximation[[2, 1]] - 36.0).abs() < epsilon);
    assert!((matrix_approximation[[2, 2]] - 81.0).abs() < epsilon);

    assert!((svd.s[0] - 123.676578742544).abs() < epsilon);
    assert!((svd.s[1] - 6.084527896514).abs() < epsilon);
    assert!((svd.s[2] - 0.287038004183).abs() < epsilon);
}
```

### Output

```text
svd.d = 3

U =
[[-0.4152068408862081, -0.7534435856189199, -0.5098294249756481],
 [-0.556377565193878, -0.23308021364108839, 0.7975698207417085],
 [-0.719755016814907, 0.6148140997884891, -0.3224226084985998]], shape=[3, 3], strides=[1, 3], layout=Ff (0xa), const ndim=2

S =
[123.67657874254405, 6.084527896513759, 0.2870380041828973], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1

V =
[[-0.07372869095916511, 0.6323518477280158, -0.7711648467120451],
 [-0.3756889918994792, 0.6986910001499903, 0.6088420712097343],
 [-0.9238083467337805, -0.33460727276072516, -0.18605405537270261]], shape=[3, 3], strides=[1, 3], layout=Ff (0xa), const ndim=2
```

### Result\<SvdRec\> for above example looks like this:

```text
svd = Ok(
    SvdRec {
       d: 3,
       ut: [[-0.4152068408862081, -0.556377565193878, -0.719755016814907],
            [-0.7534435856189199, -0.23308021364108839, 0.6148140997884891],
            [-0.5098294249756481, 0.7975698207417085, -0.3224226084985998]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2,
       s: [123.67657874254405, 6.084527896513759, 0.2870380041828973], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1,
       vt: [[-0.07372869095916511, -0.3756889918994792, -0.9238083467337805],
            [0.6323518477280158, 0.6986910001499903, -0.33460727276072516],
            [-0.7711648467120451, 0.6088420712097343, -0.18605405537270261]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2,
        diagnostics: Diagnostics {
            non_zero: 9,
            dimensions: 3,
            iterations: 3,
            transposed: false,
            lanczos_steps: 3,
            ritz_values_stabilized: 3,
            significant_values: 3,
            singular_values: 3,
            random_seed: 3141,
        },
    },
)
```
