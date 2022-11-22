/*
 * module: error
 */

use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum SvdLibError {
    #[error("svdlibrs/imtqlb: {0}")]
    ImtqlbError(String),

    #[error("svdlibrs/startv: {0}")]
    StartvError(String),

    #[error("svdlibrs/stpone: {0}")]
    StponeError(String),

    #[error("svdlibrs/imtql2: {0}")]
    Imtql2Error(String),

    #[error("svdlibrs/svdLas2: {0}")]
    Las2Error(String),

    #[error("svdlibrs/ndarray: {0}")]
    NDArrayError(#[from] ndarray::ShapeError),
}
